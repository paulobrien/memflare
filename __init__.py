"""Memflare — Hermes Agent memory provider backed by Cloudflare Agent Memory.

Activate with:
    hermes memory setup            # interactive
    hermes config set memory.provider memflare

Isolation model: memory is scoped per PERSON. When Hermes supplies a gateway
user identity (Telegram/Discord/Slack sessions), the Cloudflare profile is
derived as "<base>:<user_id>" so participants in shared chats never read or
write each other's memory. Without a user identity (single-user CLI), the
configured base profile is used directly.
"""

import json
import os
import re
import threading

try:
    from .client import (
        LIMITS,
        MemflareClient,
        MemflareError,
        sanitize_profile_component,
    )
    from .schemas import ALL_TOOLS
except ImportError:  # loaded flat (tests / direct execution) rather than as a package
    from client import LIMITS, MemflareClient, MemflareError, sanitize_profile_component
    from schemas import ALL_TOOLS

try:
    from agent.memory_provider import MemoryProvider
except ImportError:  # outside a Hermes runtime (tests, tooling)
    MemoryProvider = object

CONFIG_FILENAME = "memflare.json"
FLUSH_THRESHOLD_MESSAGES = 12

# Writes are disabled outside primary agent contexts: cron system prompts and
# subagent chatter would corrupt the user's memory representation.
NON_WRITE_CONTEXTS = frozenset({"cron", "flush", "subagent"})

SYSTEM_PROMPT_BLOCK = (
    "Long-term memory (Cloudflare Agent Memory) is active.\n"
    "- Use memory_recall when the request depends on prior sessions, preferences, "
    "project state, decisions, or conventions.\n"
    "- Do not recall what is already in the current conversation.\n"
    "- Treat recalled memory as useful context, not guaranteed truth; confirm before "
    "irreversible or high-impact actions.\n"
    "- Use memory_remember only for durable preferences, decisions, reusable "
    "instructions, and important facts. Conversation turns are ingested automatically."
)

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)


def _redact(text):
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted-secret]", text)
    return text


class MemflareMemoryProvider(MemoryProvider):
    def __init__(self):
        self._client = None
        self._base_profile = "hermes"
        self._profile = "hermes"
        self._session_id = ""
        self._hermes_home = None
        self._write_enabled = True
        # Buffer entries are (session_id, message) captured at append time, so a
        # flush can never tag another conversation's turns with the wrong session.
        self._buffer = []
        self._lock = threading.Lock()
        self._flush_thread = None
        # queue_prefetch() populates this cache in the background; prefetch()
        # consumes it on the next turn so the hot path stays fast.
        self._prefetch_query = None
        self._prefetch_result = None
        self._prefetch_thread = None

    # -- identity / availability --------------------------------------------

    @property
    def name(self):
        return "memflare"

    def is_available(self, **kwargs):
        """Config presence only — must not make network calls."""
        config = self._load_config(self._resolve_hermes_home(kwargs.get("hermes_home")))
        return bool(
            (config.get("account_id") or os.environ.get("CLOUDFLARE_ACCOUNT_ID"))
            and os.environ.get("CLOUDFLARE_API_TOKEN")
            and (config.get("namespace") or os.environ.get("CLOUDFLARE_AGENT_MEMORY_NAMESPACE"))
        )

    def _resolve_hermes_home(self, explicit=None):
        """is_available() can be called before initialize(), so fall back to the
        HERMES_HOME env var / default path when Hermes has not handed us one yet."""
        return (
            explicit
            or self._hermes_home
            or os.environ.get("HERMES_HOME")
            or os.path.expanduser("~/.hermes")
        )

    # -- configuration -------------------------------------------------------

    def get_config_schema(self):
        return [
            {
                "key": "account_id",
                "description": "Cloudflare account ID",
                "required": True,
            },
            {
                "key": "api_token",
                "description": "Cloudflare API token with Agent Memory permissions",
                "secret": True,
                "required": True,
                "env_var": "CLOUDFLARE_API_TOKEN",
                "url": "https://dash.cloudflare.com/profile/api-tokens",
            },
            {
                "key": "namespace",
                "description": (
                    "Agent Memory namespace NAME, e.g. hermes-prod — not the "
                    "namespace_id shown by wrangler (max 32 chars)"
                ),
                "required": True,
            },
            {
                "key": "profile",
                "description": (
                    "Base memory profile. Gateway users are isolated automatically "
                    "as <profile>:<user_id>; single-user CLI uses this value directly"
                ),
                "default": "hermes",
                "required": False,
            },
        ]

    def save_config(self, values, hermes_home):
        config = self._load_config(hermes_home)
        for key in ("account_id", "namespace", "profile"):
            if values.get(key):
                config[key] = values[key]
        path = os.path.join(hermes_home, CONFIG_FILENAME)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

    def initialize(self, session_id="", **kwargs):
        self._hermes_home = self._resolve_hermes_home(kwargs.get("hermes_home"))
        self._session_id = self._clip_session(session_id)
        self._write_enabled = kwargs.get("agent_context", "primary") not in NON_WRITE_CONTEXTS
        config = self._load_config(self._hermes_home)
        self._base_profile = config.get("profile") or "hermes"
        self._profile = self._resolve_profile(config, kwargs)
        self._client = MemflareClient(
            account_id=config.get("account_id") or os.environ.get("CLOUDFLARE_ACCOUNT_ID"),
            api_token=os.environ.get("CLOUDFLARE_API_TOKEN") or config.get("api_token"),
            namespace=config.get("namespace")
            or os.environ.get("CLOUDFLARE_AGENT_MEMORY_NAMESPACE"),
        )

    def _resolve_profile(self, config, kwargs):
        """Per-person isolation: '<base>:<user_id>' when Hermes supplies a
        gateway user identity, the base profile alone otherwise (CLI)."""
        user_id = kwargs.get("user_id") or kwargs.get("user_id_alt")
        if not user_id:
            return self._base_profile
        component = sanitize_profile_component(
            user_id,
            max_chars=LIMITS["profile_name_chars"] - len(self._base_profile) - 1,
        )
        return f"{self._base_profile}:{component}"

    def _load_config(self, hermes_home):
        if not hermes_home:
            return {}
        path = os.path.join(hermes_home, CONFIG_FILENAME)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _clip_session(session_id):
        return (str(session_id or ""))[: LIMITS["session_id_chars"]]

    # -- prompt / recall hooks ------------------------------------------------

    def system_prompt_block(self):
        return SYSTEM_PROMPT_BLOCK

    def prefetch(self, query, *, session_id="", **kwargs):
        with self._lock:
            cached_query, cached = self._prefetch_query, self._prefetch_result
            self._prefetch_query = self._prefetch_result = None
        if cached is not None and cached_query == query:
            return cached
        return self._recall_quietly(query)

    def queue_prefetch(self, query, *, session_id="", **kwargs):
        """Warm the recall cache for the NEXT turn in a background thread."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return

        def _warm():
            result = self._recall_quietly(query)
            with self._lock:
                self._prefetch_query, self._prefetch_result = query, result

        self._prefetch_thread = threading.Thread(target=_warm, daemon=True)
        self._prefetch_thread.start()

    def _recall_quietly(self, query):
        try:
            result = self._client.recall(self._profile, query, response_length="short")
            return (result or {}).get("answer") or ""
        except Exception:
            return ""

    # -- turn sync (must be non-blocking) --------------------------------------

    def sync_turn(self, user, assistant, *, session_id="", messages=None, **kwargs):
        if not self._write_enabled:
            return
        turn_session = self._clip_session(session_id) or self._session_id
        with self._lock:
            if user:
                self._buffer.append((turn_session, {"role": "user", "content": _redact(str(user))}))
            if assistant:
                self._buffer.append(
                    (turn_session, {"role": "assistant", "content": _redact(str(assistant))})
                )
            should_flush = len(self._buffer) >= FLUSH_THRESHOLD_MESSAGES
        if should_flush:
            self._flush_async()

    def on_session_switch(self, new_session_id, *, parent_session_id="", reset=False,
                          rewound=False, **kwargs):
        """Flush turns buffered for the old session, then adopt the new session ID
        so subsequent writes land in the correct session's record."""
        self._flush()
        self._session_id = self._clip_session(new_session_id) or self._session_id
        with self._lock:
            self._prefetch_query = self._prefetch_result = None

    def on_session_end(self, messages=None, **kwargs):
        self._flush()

    def on_pre_compress(self, messages=None, **kwargs):
        """Context compression is a natural checkpoint — flush buffered turns so
        nothing is lost when the transcript is summarized away."""
        self._flush()
        return ""

    def on_memory_write(self, action, target, content, metadata=None, **kwargs):
        """Mirror built-in MEMORY.md/USER.md writes into Cloudflare."""
        if not self._write_enabled or action not in ("add", "replace") or not content:
            return
        thread = threading.Thread(
            target=self._remember_quietly, args=(content,), daemon=True,
        )
        thread.start()

    def shutdown(self, **kwargs):
        self._flush()

    def _remember_quietly(self, content):
        try:
            self._client.remember(self._profile, _redact(str(content)),
                                  session_id=self._session_id or None)
        except Exception:
            pass

    def _flush_async(self):
        if self._flush_thread and self._flush_thread.is_alive():
            return
        self._flush_thread = threading.Thread(target=self._flush, daemon=True)
        self._flush_thread.start()

    def _flush(self):
        if self._client is None:
            return  # not initialized — leave the buffer intact
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return
        # Group by the session captured at append time — interleaved sessions
        # each ingest under their own session ID.
        groups = {}
        for session_id, message in batch:
            groups.setdefault(session_id, []).append(message)
        failed = []
        for session_id, messages in groups.items():
            try:
                for start in range(0, len(messages), LIMITS["messages_per_ingest"]):
                    chunk = messages[start:start + LIMITS["messages_per_ingest"]]
                    self._client.ingest(self._profile, chunk, session_id=session_id or None)
            except Exception:
                failed.extend((session_id, message) for message in messages)
        if failed:
            with self._lock:
                self._buffer = failed + self._buffer  # keep for the next checkpoint

    # -- tools -----------------------------------------------------------------

    def get_tool_schemas(self):
        return list(ALL_TOOLS)

    def handle_tool_call(self, tool_name, args, **kwargs):
        args = args or {}
        try:
            if self._client is None:
                return json.dumps({"error": "Memflare is not initialized. Run: hermes memory setup"})
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                return json.dumps({"error": f"Unknown memflare tool: {tool_name}"})
            return json.dumps(handler(args))
        except MemflareError as error:
            return json.dumps({"error": str(error), "status": error.status})
        except Exception as error:  # never raise into the agent loop
            return json.dumps({"error": f"Unexpected memflare failure: {error}"})

    def _tool_memory_recall(self, args):
        return self._client.recall(
            self._profile,
            args.get("query", ""),
            thinking_level=args.get("thinking_level"),
            response_length=args.get("response_length"),
        )

    def _tool_memory_list(self, args):
        return self._client.list_memories(
            self._profile,
            session_id=args.get("session_id"),
            memory_type=args.get("type"),
            per_page=args.get("per_page", 50),
            cursor=args.get("cursor"),
        )

    def _tool_memory_get(self, args):
        return self._client.get_memory(self._profile, args.get("memory_id", ""))

    def _tool_memory_summary(self, args):
        return self._client.get_summary(self._profile, session_id=args.get("session_id"))

    def _tool_memory_remember(self, args):
        result = self._client.remember(
            self._profile,
            _redact(args.get("content", "")),
            session_id=self._session_id or None,
        )
        return {"remembered": True, "result": result}

    def _tool_memory_delete(self, args):
        self._client.delete_memory(self._profile, args.get("memory_id", ""))
        return {"deleted": True, "memory_id": args.get("memory_id")}


def register(ctx):
    """Hermes plugin entry point."""
    ctx.register_memory_provider(MemflareMemoryProvider())
