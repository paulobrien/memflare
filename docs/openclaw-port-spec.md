# OpenClaw Port Specification (cfam-openclaw)

Verified platform contract for porting cfam-hermes-agent to OpenClaw, extracted from
full reads of `openclaw/extensions/memory-core`, `openclaw/extensions/memory-lancedb`,
and `openclaw/src/plugin-sdk` + `src/plugins` (source as of 2026-07, plugin API
`>=2026.6.11`). memory-lancedb is the reference scaffold.

## Platform model

- **Exclusive memory slot**, exactly like Hermes's single-provider rule. Plugins
  declare `kind: "memory"` in `definePluginEntry({...})`; the active plugin is
  selected via `config.plugins.slots.memory` (default `"memory-core"`), and the
  host auto-disables competing memory plugins (`src/plugins/slots.ts:68`).
- **Entry shape**: `definePluginEntry({ id, name, description, kind: "memory",
  configSchema, register(api: OpenClawPluginApi) })`, plus an
  `openclaw.plugin.json` manifest (id, kind, `contracts.tools`, JSON-Schema
  `configSchema`, `uiHints` with `sensitive: true` for secrets) and `package.json`
  with an `openclaw.extensions` entry-point array. TypeScript, ESM, TypeBox schemas.
- **Memory capability registration**: `api.registerMemoryCapability({ promptBuilder?,
  flushPlanResolver?, runtime?, publicArtifacts? })` — the slot owner's registration
  wins (`src/plugins/memory-state.ts:181`).

## Required tools

A replacement memory plugin must provide the exact tool names/schemas memory-core owns:

- `memory_search` — `{ query: string; maxResults?: int; minScore?: number; corpus?: "memory"|"wiki"|"all"|"sessions" }`
- `memory_get` — `{ path: string; from?: int; lines?: int; corpus? }`

Plus (memory-lancedb precedent, optional but expected of a backend plugin):

- `memory_recall` — `{ query, limit? }` → `AgentToolResult` with
  `content: [{type:"text", text}]` + `details: { count, memories[] }`
- `memory_store` — `{ text, importance? (0-1), category? }` → details
  `action: "created"|"duplicate"|"rejected"` (rejected for prompt injection)
- `memory_forget` — `{ query? | memoryId? }` → ambiguous matches return
  `action: "candidates"` with a pick-list instead of deleting (port this UX back?)

Tools return `AgentToolResult` objects (not JSON strings — differs from Hermes).
Unavailable state: `details: { disabled: true, unavailable: true, error }`.

## Hooks (via `api.registerHook(name, handler)`)

| OpenClaw hook | Hermes equivalent | Notes |
| --- | --- | --- |
| `before_prompt_build` → return `{ prependContext?, appendContext?, prependSystemContext?, appendSystemContext?, systemPrompt? }` | `prefetch` + `system_prompt_block` | Static memory policy goes in `prependSystemContext` (provider prompt-cacheable); dynamic recall in `prependContext` (per-turn cost) |
| `agent_end` — `{ runId?, messages, success, error?, durationMs? }` | `sync_turn` | Capture point; memory-lancedb uses per-session cursors (`nextIndex` + message fingerprint) to avoid reprocessing |
| `session_end` — `{ sessionId, sessionKey?, messageCount, reason: "new"\|"reset"\|"idle"\|"daily"\|"compaction"\|"deleted"\|"shutdown"\|"restart"\|"unknown", sessionFile?, nextSessionId? }` | `on_session_end` + `on_session_switch` | `reason` + `nextSession*` give richer switch semantics than Hermes |
| `before_compaction` / `after_compaction` | `on_pre_compress` | Ingest-at-compaction checkpoint — matches Cloudflare's blog guidance exactly |
| `before_message_write` — `{ message, sessionKey?, agentId? }`, can block/mutate | (none) | Per-message persistence hook; NOT a MEMORY.md hook |
| Full list | | ~40 hooks incl. `session_start`, `subagent_*`, `message_received`, `llm_input/output` (`src/plugins/hook-types.ts:81`) |

## Identity & per-person isolation (the key question — GOOD news)

`PluginHookAgentContext` (`src/plugins/hook-types.ts:243`) rides on agent hooks and carries:

```
agentId, sessionKey, sessionId, workspaceDir,
channel      // channel plugin id, e.g. "discord"
chatId       // group/conversation id
senderId     // user id within the channel
channelContext.sender  // channel-augmented sender details
```

So per-person profile derivation ports directly:
`profile = <base>-<sanitize(channel)>-<sanitize(senderId)>`, falling back to the
base profile when no channel identity exists (local single-user use). Same
collision-proof hash-suffix sanitization as the Hermes plugin. Caveats:

- No cross-app user id (`senderExternalId` deprecated/unpopulated) — identity is
  per-channel, so the same human on Discord and WhatsApp is two profiles.
- memory-lancedb itself ignores all of this (single shared table) — we would be
  ahead of the reference implementation, as we are on Hermes.
- Subagent gating: use `subagent_*` hooks / run context rather than Hermes's
  `agent_context` kwarg.

## What does NOT port

- **`on_memory_write` mirroring / end-to-end forget**: no hook observes MEMORY.md
  writes. Options: snapshot-diff MEMORY.md at `session_end`, or an fs watcher.
  Defer to v2; document the gap.
- **JSON-string tool returns / never-raise**: OpenClaw wants structured
  `AgentToolResult`s; errors surface via `details.error` + human text content.
- **Hermes config wizard**: replaced by `configSchema` + `uiHints`
  (`sensitive: true` for the API token) + `api.pluginConfig`, with live runtime
  overrides via `api.runtime.config?.current()` re-parsed inside every hook.

## What Cloudflare replaces from memory-lancedb

Delete wholesale when porting from the lancedb scaffold: both `Embeddings`
implementations, LanceDB runtime, client-side dedup (similarity > 0.95), category
detection, vector search. Cloudflare Agent Memory does extraction, dedup,
classification, and fused retrieval server-side. Keep: the sanitization pipeline
(`sanitizeForMemoryCapture`, `looksLikeEnvelopeSludge`, `looksLikePromptInjection`,
`escapeMemoryForPrompt` — HTML-escape recalled text before prompt injection),
cursor tracking, timeout wrappers (15s), and the CLI (`ltm`-style: list/search/stats).

## Port architecture

```
cfam-openclaw/
├── openclaw.plugin.json   # kind: memory, contracts.tools, configSchema, uiHints
├── package.json           # openclaw.extensions: ["./index.ts"]
├── index.ts               # definePluginEntry + register(api): tools, hooks, service, CLI
├── api.ts                 # re-exports from openclaw/plugin-sdk (convention)
├── cfam-client.ts         # TS port of client.py: endpoints, limits, charset rules,
│                          #   namespace-ID rejection, retry policy (no 409, no remember replay)
├── config.ts              # schema: accountId, apiToken (sensitive), namespace, baseProfile,
│                          #   autoRecall (default on), autoCapture (default on), flush thresholds
└── sanitize.ts            # injection/envelope filters (adapted from memory-lancedb + our patterns)
```

Wiring: `before_prompt_build` → CFAM `recall` (short) → `prependContext` (escaped,
injection-filtered); policy text → `prependSystemContext`. `agent_end` → cursor-tracked
extraction of user/assistant turns → buffered → CFAM `ingest` keyed by
per-person profile + `sessionKey`. `session_end`/`before_compaction` → flush.
`memory_search` → CFAM recall (map `maxResults`/`corpus`); `memory_get` → CFAM
`get_memory` by id (path param repurposed or corpus="memory" listing);
`memory_store`/`memory_forget` → remember/delete with our fail-safe + candidates UX.

## Effort

~3–5 days: 1 day client port, 1 day plugin shell + tools, 1 day hooks/identity,
1–2 days testing against a live OpenClaw install + docs. All Cloudflare-side
learning (charset, limits, retry, idempotency) transfers unchanged.
