"""Minimal, dependency-free HTTP client for the Cloudflare Agent Memory API.

Endpoints, field casing, and limits follow the public beta docs at
https://developers.cloudflare.com/agent-memory/. The API is in private beta;
re-verify behavior against the docs before production rollout.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

CLOUDFLARE_API_BASE_URL = "https://api.cloudflare.com/client/v4"

LIMITS = {
    "messages_per_ingest": 500,
    "message_content_bytes": 32_768,
    "recall_query_bytes": 1_024,
    "session_id_chars": 64,
    "profile_name_chars": 100,
    "namespace_name_chars": 32,
    "list_page_size_min": 1,
    "list_page_size_max": 1_000,
}

MEMORY_TYPES = ("fact", "event", "instruction", "task")
MESSAGE_ROLES = ("system", "user", "assistant")

# 409 is a semantic conflict, not transient; never retried.
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class MemflareError(Exception):
    def __init__(self, message, status=None, code=None, errors=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.errors = errors or []

    @property
    def is_not_found(self):
        return self.status == 404

    @property
    def is_conflict(self):
        return self.status == 409

    @property
    def is_retryable(self):
        return self.status is not None and (
            self.status in RETRYABLE_STATUS_CODES or self.status >= 500
        )


def _require_str(value, field, max_chars=None, max_bytes=None):
    if not isinstance(value, str) or not value.strip():
        raise MemflareError(f"{field} must be a non-empty string.")
    if max_chars is not None and len(value) > max_chars:
        raise MemflareError(f"{field} must be {max_chars} characters or fewer.")
    if max_bytes is not None and len(value.encode("utf-8")) > max_bytes:
        raise MemflareError(f"{field} must be {max_bytes} UTF-8 bytes or fewer.")
    return value


# ULID shape of Cloudflare namespace_id values — a common paste mistake, since
# the API addresses namespaces by NAME, not ID.
_NAMESPACE_ID_SHAPE = re.compile(r"^[0-9][0-9A-HJKMNP-TV-Za-hjkmnp-tv-z]{25}$")


def validate_namespace(name):
    _require_str(name, "namespace", max_chars=LIMITS["namespace_name_chars"])
    if _NAMESPACE_ID_SHAPE.match(name):
        raise MemflareError(
            f"namespace '{name}' looks like a namespace_id. Use the namespace NAME "
            "(e.g. hermes-prod) — the Cloudflare API addresses namespaces by name."
        )
    return name


def validate_profile(name):
    return _require_str(name, "profile", max_chars=LIMITS["profile_name_chars"])


def sanitize_profile_component(value, max_chars=LIMITS["profile_name_chars"]):
    """Make an arbitrary identifier (gateway user ID, etc.) safe for use inside
    a profile name. Any identifier sanitization ALTERS gets a stable hash suffix
    so distinct raw IDs (e.g. 'user 1' vs 'user@1') can never collapse into the
    same profile."""
    raw = str(value or "").strip()
    normalized = re.sub(r"[^A-Za-z0-9._:-]+", "-", raw).strip("-")
    if not normalized:
        raise MemflareError("profile component must contain usable characters.")
    suffix = "-" + _fnv1a_hex(raw)[:8]
    if max_chars <= len(suffix):
        return _fnv1a_hex(raw)[:max(max_chars, 1)]
    if normalized == raw and len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - len(suffix)] + suffix


def _fnv1a_hex(value):
    hash_value = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{hash_value:016x}"


def validate_session_id(session_id):
    if session_id is None or session_id == "":
        return None
    return _require_str(session_id, "sessionId", max_chars=LIMITS["session_id_chars"])


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise MemflareError("messages must be a non-empty list.")
    if len(messages) > LIMITS["messages_per_ingest"]:
        raise MemflareError(
            f"messages must contain {LIMITS['messages_per_ingest']} or fewer per ingest call."
        )
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise MemflareError(f"messages[{index}] must be an object.")
        role = message.get("role")
        if role not in MESSAGE_ROLES:
            raise MemflareError(f"messages[{index}].role must be one of {MESSAGE_ROLES}.")
        content = _require_str(
            message.get("content"),
            f"messages[{index}].content",
            max_bytes=LIMITS["message_content_bytes"],
        )
        entry = {"role": role, "content": content}
        if message.get("timestamp"):
            entry["timestamp"] = str(message["timestamp"])
        normalized.append(entry)
    return normalized


class MemflareClient:
    def __init__(self, account_id, api_token, namespace,
                 base_url=CLOUDFLARE_API_BASE_URL, timeout=15.0, retries=2):
        self.account_id = _require_str(account_id, "account_id")
        self.api_token = _require_str(api_token, "api_token")
        self.namespace = validate_namespace(namespace)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = 1.0

    # -- namespaces ---------------------------------------------------------

    def create_namespace(self, name=None):
        name = validate_namespace(name or self.namespace)
        return self._request("POST", "/namespaces", body={"name": name}, retryable=False)

    def get_namespace(self, name=None):
        name = validate_namespace(name or self.namespace)
        return self._request("GET", f"/namespaces/{_seg(name)}")

    def ensure_namespace(self, name=None):
        name = validate_namespace(name or self.namespace)
        try:
            return self.get_namespace(name)
        except MemflareError as error:
            if not error.is_not_found:
                raise
        try:
            return self.create_namespace(name)
        except MemflareError as error:
            if error.is_conflict:
                return self.get_namespace(name)
            raise

    # -- profile operations -------------------------------------------------

    def ingest(self, profile, messages, session_id=None):
        body = {"messages": normalize_messages(messages)}
        session_id = validate_session_id(session_id)
        if session_id:
            body["sessionId"] = session_id
        return self._request("POST", self._profile_path(profile, "/ingest"), body=body)

    def remember(self, profile, content, session_id=None):
        body = {"content": _require_str(content, "content")}
        session_id = validate_session_id(session_id)
        if session_id:
            body["sessionId"] = session_id
        return self._request(
            "POST", self._profile_path(profile, "/remember"), body=body, retryable=False,
        )

    def recall(self, profile, query, thinking_level=None, response_length=None,
               reference_date=None):
        body = {"query": _require_str(query, "query", max_bytes=LIMITS["recall_query_bytes"])}
        if thinking_level:
            if thinking_level not in ("low", "medium", "high"):
                raise MemflareError("thinkingLevel must be low, medium, or high.")
            body["thinkingLevel"] = thinking_level
        if response_length:
            if response_length not in ("short", "medium", "long"):
                raise MemflareError("responseLength must be short, medium, or long.")
            body["responseLength"] = response_length
        if reference_date:
            body["referenceDate"] = str(reference_date)
        return self._request("POST", self._profile_path(profile, "/recall"), body=body)

    def list_memories(self, profile, session_id=None, memory_type=None,
                      per_page=50, cursor=None):
        if not isinstance(per_page, int) or not (
            LIMITS["list_page_size_min"] <= per_page <= LIMITS["list_page_size_max"]
        ):
            raise MemflareError(
                f"per_page must be an integer between {LIMITS['list_page_size_min']} "
                f"and {LIMITS['list_page_size_max']}."
            )
        if memory_type is not None and memory_type not in MEMORY_TYPES:
            raise MemflareError(f"type must be one of {MEMORY_TYPES}.")
        query = {"per_page": per_page}
        if validate_session_id(session_id):
            query["session_id"] = session_id
        if memory_type:
            query["type"] = memory_type
        if cursor:
            query["cursor"] = cursor
        return self._request(
            "GET", self._profile_path(profile, "/memories"), query=query, with_info=True,
        )

    def get_memory(self, profile, memory_id):
        memory_id = _require_str(memory_id, "memory_id")
        return self._request("GET", self._profile_path(profile, f"/memories/{_seg(memory_id)}"))

    def delete_memory(self, profile, memory_id):
        memory_id = _require_str(memory_id, "memory_id")
        return self._request(
            "DELETE", self._profile_path(profile, f"/memories/{_seg(memory_id)}"),
        )

    def delete_session(self, profile, session_id):
        session_id = validate_session_id(session_id)
        if not session_id:
            raise MemflareError("session_id is required.")
        return self._request(
            "DELETE", self._profile_path(profile, f"/sessions/{_seg(session_id)}"),
        )

    def get_summary(self, profile, session_id=None):
        body = {}
        session_id = validate_session_id(session_id)
        if session_id:
            body["sessionId"] = session_id
        return self._request("POST", self._profile_path(profile, "/summary"), body=body)

    # -- transport ----------------------------------------------------------

    def _profile_path(self, profile, suffix):
        profile = validate_profile(profile)
        return f"/namespaces/{_seg(self.namespace)}/profiles/{_seg(profile)}{suffix}"

    def _request(self, method, path, body=None, query=None, retryable=True, with_info=False):
        url = f"{self.base_url}/accounts/{_seg(self.account_id)}/agent-memory{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"

        max_attempts = (self.retries if retryable else 0) + 1
        last_error = None
        for attempt in range(max_attempts):
            try:
                status, payload = self._send(method, url, headers, data)
                if status >= 400 or payload.get("success") is False:
                    errors = payload.get("errors") or []
                    message = errors[0].get("message") if errors else f"HTTP {status}"
                    raise MemflareError(
                        f"Cloudflare Agent Memory request failed: {message}",
                        status=status,
                        code=errors[0].get("code") if errors else None,
                        errors=errors,
                    )
                if with_info:
                    return {
                        "result": payload.get("result"),
                        "result_info": payload.get("result_info"),
                    }
                return payload.get("result")
            except MemflareError as error:
                last_error = error
                if attempt >= max_attempts - 1 or (error.status is not None and not error.is_retryable):
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last_error = MemflareError(f"Cloudflare Agent Memory request failed: {error}")
                if attempt >= max_attempts - 1:
                    raise last_error from error
            time.sleep(min(self.retry_backoff * 2 ** attempt, 8.0))
        raise last_error

    def _send(self, method, url, headers, data):
        """Single HTTP round trip. Tests replace this method."""
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, _parse_json(response.read())
        except urllib.error.HTTPError as error:
            return error.code, _parse_json(error.read())


def _parse_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {"success": False, "errors": [{"message": "Non-JSON response from API."}]}


def _seg(value):
    return urllib.parse.quote(str(value), safe="")
