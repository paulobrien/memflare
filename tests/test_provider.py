import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import __init__ as memflare  # noqa: E402  (plugin package loaded flat for tests)


class FakeClient:
    def __init__(self):
        self.calls = []

    def recall(self, profile, query, **kwargs):
        self.calls.append(("recall", profile, query, kwargs))
        return {"answer": "You prefer concise answers.", "count": 1}

    def remember(self, profile, content, session_id=None):
        self.calls.append(("remember", profile, content, session_id))
        return {"id": "mem-1"}

    def ingest(self, profile, messages, session_id=None):
        self.calls.append(("ingest", profile, messages, session_id))

    def list_memories(self, profile, **kwargs):
        self.calls.append(("list", profile, kwargs))
        return {"result": [], "result_info": {}}

    def get_memory(self, profile, memory_id):
        return {"id": memory_id}

    def delete_memory(self, profile, memory_id):
        self.calls.append(("delete", profile, memory_id))

    def get_summary(self, profile, session_id=None):
        return {"summary": "# Memory"}


def make_provider():
    provider = memflare.MemflareMemoryProvider()
    provider._client = FakeClient()
    provider._profile = "hermes"
    provider._session_id = "sess-1"
    return provider


class ProviderTests(unittest.TestCase):
    def test_handle_tool_call_returns_json_string(self):
        provider = make_provider()
        raw = provider.handle_tool_call("memory_recall", {"query": "preferences"})
        self.assertIsInstance(raw, str)
        self.assertEqual(json.loads(raw)["answer"], "You prefer concise answers.")

    def test_handle_tool_call_never_raises(self):
        provider = make_provider()

        def boom(*args, **kwargs):
            raise RuntimeError("kaput")

        provider._client.recall = boom
        raw = provider.handle_tool_call("memory_recall", {"query": "x"})
        self.assertIn("error", json.loads(raw))

        raw = provider.handle_tool_call("nonexistent_tool", {})
        self.assertIn("Unknown memflare tool", json.loads(raw)["error"])

        uninitialized = memflare.MemflareMemoryProvider()
        raw = uninitialized.handle_tool_call("memory_recall", {"query": "x"})
        self.assertIn("not initialized", json.loads(raw)["error"])

    def test_tool_schemas_are_flat_hermes_format(self):
        provider = make_provider()
        for schema in provider.get_tool_schemas():
            self.assertIn("name", schema)
            self.assertIn("description", schema)
            self.assertEqual(schema["parameters"]["type"], "object")
            self.assertNotIn("function", schema)

    def test_sync_turn_buffers_and_flushes_at_threshold(self):
        provider = make_provider()
        for i in range(memflare.FLUSH_THRESHOLD_MESSAGES // 2):
            provider.sync_turn(f"question {i}", f"answer {i}")
        if provider._flush_thread:
            provider._flush_thread.join(timeout=5)

        ingests = [c for c in provider._client.calls if c[0] == "ingest"]
        self.assertEqual(len(ingests), 1)
        self.assertEqual(len(ingests[0][2]), memflare.FLUSH_THRESHOLD_MESSAGES)
        self.assertEqual(ingests[0][3], "sess-1")
        self.assertEqual(len(provider._buffer), 0)

    def test_failed_flush_requeues_messages(self):
        provider = make_provider()

        def failing_ingest(*args, **kwargs):
            raise memflare.MemflareError("boom", status=503)

        provider._client.ingest = failing_ingest
        provider.sync_turn("hello", "hi")
        provider.on_session_end()
        self.assertEqual(len(provider._buffer), 2)

    def test_redaction_scrubs_secrets(self):
        provider = make_provider()
        provider.sync_turn("my key is sk-abcdefghijklmnop1234", None)
        self.assertIn("[redacted-secret]", provider._buffer[0]["content"])

    def test_on_memory_write_mirrors_to_remember(self):
        provider = make_provider()
        provider.on_memory_write("add", "MEMORY.md", "User prefers dark mode.")
        for thread in list(__import__("threading").enumerate()):
            if thread.daemon and thread.name.startswith("Thread"):
                thread.join(timeout=5)
        remembers = [c for c in provider._client.calls if c[0] == "remember"]
        self.assertEqual(len(remembers), 1)
        self.assertEqual(remembers[0][2], "User prefers dark mode.")

    def test_prefetch_fails_silently(self):
        provider = make_provider()

        def boom(*args, **kwargs):
            raise RuntimeError("offline")

        provider._client.recall = boom
        self.assertEqual(provider.prefetch("preferences"), "")

    def test_config_roundtrip_and_availability(self):
        provider = memflare.MemflareMemoryProvider()
        with tempfile.TemporaryDirectory() as home:
            provider.save_config(
                {"account_id": "acct-1", "namespace": "hermes-prod", "profile": "me"},
                home,
            )
            config = provider._load_config(home)
            self.assertEqual(config["account_id"], "acct-1")
            self.assertEqual(config["profile"], "me")

    def test_is_available_reads_config_before_initialize(self):
        provider = memflare.MemflareMemoryProvider()
        with tempfile.TemporaryDirectory() as home:
            provider.save_config({"account_id": "acct-1", "namespace": "ns"}, home)
            env = {"HERMES_HOME": home, "CLOUDFLARE_API_TOKEN": "token-1"}
            with unittest.mock.patch.dict("os.environ", env, clear=False):
                self.assertTrue(provider.is_available())
            with unittest.mock.patch.dict("os.environ", {"HERMES_HOME": home}, clear=True):
                self.assertFalse(provider.is_available())  # token missing

    def test_register_wires_provider(self):
        registered = []

        class Ctx:
            def register_memory_provider(self, provider):
                registered.append(provider)

        memflare.register(Ctx())
        self.assertEqual(len(registered), 1)
        self.assertEqual(registered[0].name, "memflare")


if __name__ == "__main__":
    unittest.main()
