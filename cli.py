"""Optional CLI: `hermes cfam-hermes-agent status` — verify config and connectivity."""

import json
import os


def _run(args):
    sub = getattr(args, "cfam_command", None)
    if sub != "status":
        print("Usage: hermes cfam-hermes-agent status")
        return

    try:
        from .client import CfamClient
    except ImportError:
        from client import CfamClient

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    namespace = os.environ.get("CLOUDFLARE_AGENT_MEMORY_NAMESPACE")

    hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    for filename in ("cfam-hermes-agent.json", "memflare.json"):
        config_path = os.path.join(hermes_home, filename)
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            account_id = config.get("account_id") or account_id
            namespace = config.get("namespace") or namespace
            break

    missing = [name for name, value in (
        ("account_id", account_id),
        ("CLOUDFLARE_API_TOKEN", token),
        ("namespace", namespace),
    ) if not value]
    if missing:
        print(f"cfam-hermes-agent: not configured (missing: {', '.join(missing)}). Run: hermes memory setup")
        return

    client = CfamClient(account_id=account_id, api_token=token, namespace=namespace)
    try:
        result = client.get_namespace()
        print(f"cfam-hermes-agent: OK — namespace '{namespace}' reachable ({json.dumps(result)})")
    except Exception as error:
        print(f"cfam-hermes-agent: configuration found but API check failed: {error}")


def register_cli(subparser):
    subs = subparser.add_subparsers(dest="cfam_command")
    subs.add_parser("status", help="Check cfam-hermes-agent configuration and Cloudflare connectivity")
    subparser.set_defaults(func=_run)
