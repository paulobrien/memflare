"""Optional CLI: `hermes memflare status` — verify config and connectivity."""

import json
import os


def _run(args):
    sub = getattr(args, "memflare_command", None)
    if sub != "status":
        print("Usage: hermes memflare status")
        return

    try:
        from .client import MemflareClient
    except ImportError:
        from client import MemflareClient

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    namespace = os.environ.get("CLOUDFLARE_AGENT_MEMORY_NAMESPACE")

    hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    config_path = os.path.join(hermes_home, "memflare.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        account_id = config.get("account_id") or account_id
        namespace = config.get("namespace") or namespace

    missing = [name for name, value in (
        ("account_id", account_id),
        ("CLOUDFLARE_API_TOKEN", token),
        ("namespace", namespace),
    ) if not value]
    if missing:
        print(f"memflare: not configured (missing: {', '.join(missing)}). Run: hermes memory setup")
        return

    client = MemflareClient(account_id=account_id, api_token=token, namespace=namespace)
    try:
        result = client.get_namespace()
        print(f"memflare: OK — namespace '{namespace}' reachable ({json.dumps(result)})")
    except Exception as error:
        print(f"memflare: configuration found but API check failed: {error}")


def register_cli(subparser):
    subs = subparser.add_subparsers(dest="memflare_command")
    subs.add_parser("status", help="Check Memflare configuration and Cloudflare connectivity")
    subparser.set_defaults(func=_run)
