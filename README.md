# cfam-hermes-agent

cfam-hermes-agent is a [Hermes Agent](https://hermes-agent.nousresearch.com/) **memory provider plugin** backed by [Cloudflare Agent Memory](https://developers.cloudflare.com/agent-memory/). It gives Hermes durable, growing memory across sessions without Hermes owning a database, vector index, summarizer, or deduplication pipeline.

Cloudflare Agent Memory is in private beta. cfam-hermes-agent follows the public beta docs; re-verify API behavior before relying on it in production.

**Full documentation lives in the [project wiki](../../wiki)** (user guide: installation, configuration, tools, troubleshooting; developer guide: architecture, provider contract, client reference).

## What it does

Once active, Hermes automatically gets:

- **Per-person memory isolation**: when Hermes supplies a gateway user identity (Telegram/Discord/Slack), memory is scoped to `<profile>-<user_id>` — participants in group chats never read or write each other's memory. Single-user CLI sessions use the configured base profile directly.
- **Memory tools** the model can call: `memory_recall`, `memory_list`, `memory_get`, `memory_summary`, `memory_remember`, `memory_delete`.
- **Automatic checkpoint ingestion**: conversation turns are buffered and ingested in the background (non-blocking), letting Cloudflare extract durable facts, events, instructions, and tasks. Cloudflare's `ingest` is idempotent, so checkpoints never create duplicates.
- **Full built-in memory mirroring, including forgetting**: adds/replaces to Hermes's own `MEMORY.md`/`USER.md` are mirrored to Cloudflare, replaces retire the superseded copy, and removals propagate as best-effort deletes — "forget" works end-to-end, which no other bundled provider does.
- **A system-prompt policy block** telling the model when to recall and what is worth remembering.
- **Secret redaction**: common API-key/token patterns are scrubbed before anything leaves the machine.

## Requirements

- Hermes Agent with plugin support
- Python 3.9+ (no third-party dependencies — stdlib only)
- A Cloudflare account with Agent Memory (private beta) enabled

## Install

**1. Create a Cloudflare Agent Memory namespace** (once per environment):

```sh
wrangler agent-memory namespace create hermes-prod
```

Wrangler's output table shows a `namespace_id` — you won't need it. cfam-hermes-agent (and the Cloudflare API) address namespaces by **name** (`hermes-prod` here), so that's the value to use everywhere below.

**2. Install the plugin** into the Hermes **memory plugins** directory (note the `memory/` category subdirectory — memory providers are not discovered from the top-level plugins folder):

```sh
git clone <this-repo-url> ~/.hermes/plugins/memory/cfam-hermes-agent
hermes plugins enable cfam-hermes-agent   # user-dir installs only; skip for in-tree
```

**3. Activate it as the memory provider:**

```sh
hermes memory setup        # interactive picker — choose cfam-hermes-agent and enter credentials
```

or configure directly:

```sh
hermes config set memory.provider cfam-hermes-agent
```

The setup wizard prompts for the fields from the config schema: your Cloudflare **account ID**, an **API token** with Agent Memory permissions (stored as `CLOUDFLARE_API_TOKEN` in the profile's `.env`), the **namespace name** (not the `namespace_id`), and an optional **profile** name (defaults to `hermes`). Non-secret values are stored in `$HERMES_HOME/cfam-hermes-agent.json`, so each Hermes profile keeps isolated memory configuration.

**4. Verify:**

```sh
hermes memory status       # shows cfam-hermes-agent as the active provider
hermes cfam-hermes-agent status     # checks Cloudflare connectivity
```

Then ask Hermes "what do you remember about me?" and watch for a `memory_recall` call.

## How memory is organized

Cloudflare Agent Memory stores data as `namespace > profile > session > memories`. cfam-hermes-agent maps:

| Cloudflare concept | cfam-hermes-agent usage |
| --- | --- |
| Namespace | One per environment (e.g. `hermes-dev`, `hermes-prod`), set in config |
| Profile | **One per person**: `<base>-<user_id>` for gateway users, the base `profile` config value (default `hermes`) for single-user CLI |
| Session | The Hermes session ID, captured per turn (interleaved chats never mis-tag) |
| Memory types | `fact`, `event`, `instruction`, `task` (Cloudflare-classified) |

Gateway user IDs are sanitized before use in profile names; any ID that sanitization alters gets a stable hash suffix so distinct raw IDs can never collapse into the same profile. Use separate namespaces for hard environment boundaries, and separate base `profile` values if multiple Hermes profiles share one namespace.

## Lifecycle hooks implemented

| Hook | Behavior |
| --- | --- |
| `system_prompt_block()` | Injects the memory policy into the system prompt |
| `prefetch(query)` | Serves the background-warmed recall cache, falling back to a live short recall; fails silently |
| `queue_prefetch(query)` | Warms the recall cache for the next turn in a daemon thread |
| `sync_turn(user, assistant, session_id=…)` | Buffers turns tagged with their session at capture time; flushes to `ingest` in a daemon thread every 12 messages (never blocks the agent loop) |
| `on_session_switch(new_session_id)` | Flushes the old session's buffered turns, then adopts the new session ID |
| `on_session_end(messages)` | Flushes any remaining buffered turns |
| `on_pre_compress(messages)` | Flushes before context compression so turns aren't lost with the compressed transcript |
| `on_memory_write(action, target, content, metadata)` | Mirrors built-in memory writes in the background: adds are remembered, replaces retire the old copy first, and removes propagate as deletes (only ever on a single confident match — ambiguity fails safe) |
| `shutdown()` | Final flush |

Writes are disabled entirely in non-primary agent contexts (`cron`, `flush`, `subagent`) — hooks *and* the `memory_remember`/`memory_delete` tools — so scheduled jobs and subagent chatter never pollute a person's memory. Failed background flushes re-queue their messages for the next checkpoint rather than dropping them, and interleaved sessions each flush under their own session ID.

## Tool handler contract

All tool handlers follow the Hermes plugin rules: `handle_tool_call(tool_name, args, **kwargs)` always returns a JSON **string** and never raises — validation problems, API errors, and unexpected failures come back as `{"error": ...}` for the model to read.

`memory_remember` and `memory_delete` are deliberately conservative: their descriptions instruct the model to store only durable knowledge and to delete only on explicit user request. Turn-by-turn capture happens through checkpoint ingestion, not model-initiated writes.

## Limits enforced client-side

cfam-hermes-agent validates the documented Cloudflare limits before making requests:

| Limit | Value |
| --- | --- |
| Messages per ingest call | 500 (larger buffers are chunked) |
| Message content size | 32,768 UTF-8 bytes |
| Recall query size | 1,024 UTF-8 bytes |
| Session ID length | 64 characters |
| Profile name length | 100 characters |
| Namespace name length | 32 characters |
| List page size | 1–1,000 |

Retry policy: transient statuses (`408`, `425`, `429`, any `5xx`) and network errors are retried with exponential backoff for safe-to-replay calls. `409` conflicts are never retried, and `remember` is never retried because replaying it after a dropped response could store duplicate explicit memories.

## Development

```sh
python -m unittest discover -s tests -v
```

Tests use a fake transport/client — no Hermes runtime and no Cloudflare credentials required. If plugin discovery misbehaves, run `HERMES_PLUGINS_DEBUG=1 hermes plugins list`.

## Source map

```txt
plugin.yaml    Plugin manifest (tools, hooks, required env)
__init__.py    MemoryProvider implementation + register(ctx) entry point
client.py      Cloudflare Agent Memory HTTP client (stdlib only)
schemas.py     Flat Hermes tool schemas
cli.py         `hermes cfam-hermes-agent status` CLI subcommand
tests/         Unit tests (unittest, no network)
```

## Official references

- [Hermes plugin guide](https://hermes-agent.nousresearch.com/docs/guides/build-a-hermes-plugin)
- [Hermes memory provider developer guide](https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin)
- [Cloudflare Agent Memory overview](https://developers.cloudflare.com/agent-memory/)
- [HTTP API](https://developers.cloudflare.com/agent-memory/api/http-api/)
- [Limits](https://developers.cloudflare.com/agent-memory/platform/limits/)
