import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import LIMITS, MemflareClient, MemflareError  # noqa: E402


def make_client(responder):
    client = MemflareClient(
        account_id="acct-123", api_token="token-123", namespace="hermes-prod",
    )
    client._send = responder
    client.retry_backoff = 0.0
    return client


def ok(result=None, result_info=None):
    payload = {"success": True, "errors": [], "result": result}
    if result_info is not None:
        payload["result_info"] = result_info
    return 200, payload


class ClientTests(unittest.TestCase):
    def test_recall_builds_correct_request(self):
        calls = []

        def responder(method, url, headers, data):
            calls.append((method, url, headers, data))
            return ok({"answer": "Concise TypeScript.", "count": 1})

        client = make_client(responder)
        result = client.recall("user-42", "coding style", thinking_level="low",
                               response_length="short")

        self.assertEqual(result["answer"], "Concise TypeScript.")
        method, url, headers, data = calls[0]
        self.assertEqual(method, "POST")
        self.assertIn(
            "/accounts/acct-123/agent-memory/namespaces/hermes-prod/profiles/user-42/recall",
            url,
        )
        self.assertEqual(headers["Authorization"], "Bearer token-123")
        self.assertEqual(
            json.loads(data),
            {"query": "coding style", "thinkingLevel": "low", "responseLength": "short"},
        )

    def test_validates_limits_before_sending(self):
        def responder(*_):
            raise AssertionError("must not send")

        client = make_client(responder)
        with self.assertRaisesRegex(MemflareError, "1024 UTF-8 bytes"):
            client.recall("user-42", "x" * (LIMITS["recall_query_bytes"] + 1))
        with self.assertRaisesRegex(MemflareError, "non-empty list"):
            client.ingest("user-42", [])

    def test_rejects_namespace_id_instead_of_name(self):
        with self.assertRaisesRegex(MemflareError, "looks like a namespace_id"):
            MemflareClient(
                account_id="acct-123",
                api_token="token-123",
                namespace="01KWPF44FYY3Q1NP2N8NX11SBB",
            )
        # A lowercased paste of the same ID is also caught.
        with self.assertRaisesRegex(MemflareError, "looks like a namespace_id"):
            MemflareClient(
                account_id="acct-123",
                api_token="token-123",
                namespace="01kwpf44fyy3q1np2n8nx11sbb",
            )

    def test_retries_transient_errors_then_succeeds(self):
        calls = {"n": 0}

        def responder(method, url, headers, data):
            calls["n"] += 1
            if calls["n"] == 1:
                return 503, {"success": False, "errors": [{"message": "unavailable"}]}
            return ok({"answer": "recovered"})

        client = make_client(responder)
        client.retries = 2
        result = client.recall("user-42", "preferences")
        self.assertEqual(result["answer"], "recovered")
        self.assertEqual(calls["n"], 2)

    def test_does_not_retry_conflicts(self):
        calls = {"n": 0}

        def responder(method, url, headers, data):
            calls["n"] += 1
            return 409, {"success": False, "errors": [{"code": 10008, "message": "Conflict"}]}

        client = make_client(responder)
        with self.assertRaises(MemflareError) as caught:
            client.recall("user-42", "preferences")
        self.assertTrue(caught.exception.is_conflict)
        self.assertEqual(calls["n"], 1)

    def test_does_not_retry_remember(self):
        calls = {"n": 0}

        def responder(method, url, headers, data):
            calls["n"] += 1
            return 500, {"success": False, "errors": [{"message": "boom"}]}

        client = make_client(responder)
        with self.assertRaises(MemflareError):
            client.remember("user-42", "The user prefers concise answers.")
        self.assertEqual(calls["n"], 1)

    def test_ensure_namespace_creates_when_missing(self):
        calls = []

        def responder(method, url, headers, data):
            calls.append((method, url))
            if method == "GET" and len(calls) == 1:
                return 404, {"success": False, "errors": [{"message": "not found"}]}
            return ok({"name": "hermes-prod"})

        client = make_client(responder)
        result = client.ensure_namespace()
        self.assertEqual(result["name"], "hermes-prod")
        self.assertEqual(calls[1][0], "POST")


if __name__ == "__main__":
    unittest.main()
