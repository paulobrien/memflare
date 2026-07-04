"""Memflare — Hermes Agent memory provider backed by Cloudflare Agent Memory.

Activate with:
    hermes memory setup            # interactive
    hermes config set memory.provider memflare
"""

import json
import os
import re
import threading

try:
    from .client import LIMITS, MemflareClient, MemflareError
    from .schemas import ALL_TOOLS
except ImportError:  # loaded flat (tests / direct execution) rather than as a package
    from client import LIMITS, MemflareClient, MemflareError
    from schemas import ALL_TOOLS

try:
    from agent.memory_provider import MemoryProvider
except ImportError:  # outside a Hermes runtime (tests, tooling)
    MemoryProvider = object

CONFIG_FILENAME = "memflare.json"
FLUSH_THRESHOLD_MESSAGES = 12

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
        self._profile = "hermes"
        self._session_id = ""
        self._hermes_home = None
        self._buffer = []
        self._lock = threading.Lock()
        self._flush_thread = None

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
                "description": "Agent Memory namespace (max 32 chars)",
                "required": True,
            },
            {
                "key": "profile",
                "description": "Memory profile name inside the namespace",
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
        self._session_id = (session_id or "")[: LIMITS["session_id_chars"]]
        config = self._load_config(self._hermes_home)
        self._profile = config.get("profile") or "hermes"
        self._client = MemflareClient(
            account_id=config.get("account_id") or os.environ.get("CLOUDFLARE_ACCOUNT_ID"),
            api_token=os.environ.get("CLOUDFLARE_API_TOKEN") or config.get("api_token"),
            namespace=config.get("namespace")
            or os.environ.get("CLOUDFLARE_AGENT_MEMORY_NAMESPACE"),
        )

    def _load_config(self, hermes_home):
        if not hermes_home:
            return {}
        path = os.path.join(hermes_home, CONFIG_FILENAME)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    # -- prompt / recall hooks ------------------------------------------------

    def system_prompt_block(self):
        return SYSTEM_PROMPT_BLOCK

    def prefetch(self, query, *, session_id="", **kwargs):
        try:
            result = self._client.recall(self._profile, query, response_length="short")
            return (result or {}).get("answer") or ""
        except Exception:
            return ""

    # -- turn sync (must be non-blocking) --------------------------------------

    def sync_turn(self, user, assistant, *, session_id="", messages=None, **kwargs):
        with self._lock:
            if user:
                self._buffer.append({"role": "user", "content": _redact(str(user))})
            if assistant:
                self._buffer.append({"role": "assistant", "content": _redact(str(assistant))})
            should_flush = len(self._buffer) >= FLUSH_THRESHOLD_MESSAGES
        if should_flush:
            self._flush_async(session_id or self._session_id)

    def on_session_end(self, messages=None, **kwargs):
        self._flush(self._session_id)

    def on_memory_write(self, action, target, content, **kwargs):
        """Mirror built-in MEMORY.md/USER.md writes into Cloudflare."""
        if action not in ("add", "replace") or not content:
            return
        thread = threading.Thread(
            target=self._remember_quietly, args=(content,), daemon=True,
        )
        thread.start()

    def shutdown(self, **kwargs):
        self._flush(self._session_id)

    def _remember_quietly(self, content):
        try:
            self._client.remember(self._profile, _redact(str(content)),
                                  session_id=self._session_id or None)
        except Exception:
            pass

    def _flush_async(self, session_id):
        if self._flush_thread and self._flush_thread.is_alive():
            return
        self._flush_thread = threading.Thread(
            target=self._flush, args=(session_id,), daemon=True,
        )
        self._flush_thread.start()

    def _flush(self, session_id):
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch or self._client is None:
            return
        try:
            for start in range(0, len(batch), LIMITS["messages_per_ingest"]):
                chunk = batch[start:start + LIMITS["messages_per_ingest"]]
                self._client.ingest(self._profile, chunk, session_id=session_id or None)
        except Exception:
            with self._lock:
                self._buffer = batch + self._buffer  # keep for the next checkpoint

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
