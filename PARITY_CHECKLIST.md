# Parity Checklist Against npm `src`

This document tracks what is already implemented in Python and what is still missing compared with the upstream npm runtime.

This is a functionality-oriented checklist, not a line-by-line source equivalence claim. Large parts of the mirrored Python workspace still act as inventory or scaffolding, while the working Python runtime currently lives mainly in [`src/agent_runtime.py`](src/agent_runtime.py), [`src/query_engine.py`](src/query_engine.py), [`src/agent_tools.py`](src/agent_tools.py), [`src/agent_prompting.py`](src/agent_prompting.py), [`src/agent_context.py`](src/agent_context.py), [`src/agent_manager.py`](src/agent_manager.py), [`src/plugin_runtime.py`](src/plugin_runtime.py), [`src/agent_slash_commands.py`](src/agent_slash_commands.py), and [`src/openai_compat.py`](src/openai_compat.py).

## 1. Core Agent Runtime

Done:

- [x] One-shot agent loop with iterative tool calling
- [x] OpenAI-compatible `chat/completions` client
- [x] Streaming token-by-token assistant output
- [x] Local-model execution against `vLLM`
- [x] Local-model execution through `Ollama`
- [x] Local-model execution through `LiteLLM Proxy`
- [x] Transcript-aware session object for the Python runtime
- [x] Session save and resume support
- [x] Configurable max-turn execution
- [x] Permission-aware tool execution
- [x] Structured output / JSON schema request mode
- [x] Cost tracking and usage budget enforcement
- [x] Scratchpad directory integration
- [x] File history journaling for write/edit/shell tool actions
- [x] Incremental `bash` tool-result streaming events
- [x] Incremental tool-result streaming for read-only text tools
- [x] Incremental tool-result streaming across the current Python text tool surface
- [x] Mutable tool transcript updates during tool execution
- [x] Transcript mutation history for replaced/tombstoned messages
- [x] Assistant streaming and tool-call transcript mutation history
- [x] Session-wide mutation serial tracking across transcript updates
- [x] Structured transcript block export for messages, tool calls, and tool results
- [x] Resume-time file-history replay reminders
- [x] Resume-time file-history snapshot previews for file edits
- [x] File-history snapshot ids and replay summaries for file edits
- [x] File-history result previews for shell and delegated-tool entries
- [x] Truncated-response continuation flow for `finish_reason=length`
- [x] Basic snipping of older tool/tool-call messages for context control
- [x] Basic automatic compact-boundary insertion with preserved recent tail
- [x] Reactive compaction retry after prompt-too-long backend failures
- [x] Reasoning-token budget enforcement
- [x] Tool-call and delegated-task budget enforcement
- [x] Resume-aware cumulative model-call budgets
- [x] Resume-aware cumulative session usage/cost persistence
- [x] Basic nested-agent delegation tool
- [x] Sequential multi-subtask delegation with parent-context carryover
- [x] Dependency-aware delegated subtasks
- [x] Topological dependency-batch delegation planning
- [x] Basic agent-manager lineage tracking for nested agents
- [x] Managed agent-group membership tracking with child indices
- [x] Agent-manager strategy and batch summary tracking for delegated groups
- [x] Delegated child-session resume by saved session id
- [x] Agent-manager tracking for resumed child-session lineage
- [x] Plugin-cache discovery and prompt-context injection
- [x] Manifest-based plugin runtime discovery
- [x] Manifest-defined plugin hooks for before-prompt and after-turn runtime injection
- [x] Manifest-defined plugin lifecycle hooks for resume, persist, and delegate phases
- [x] Manifest-defined plugin tool aliases over base runtime tools
- [x] Manifest-defined executable virtual tools
- [x] Manifest-defined plugin tool blocking
- [x] Manifest-defined plugin `beforeTool` guidance
- [x] Manifest-defined plugin tool-result guidance injected back into the transcript
- [x] Plugin runtime session-state persistence and resume restoration
- [x] Manifest-based hook/policy runtime discovery
- [x] Hook/policy before-prompt runtime injection
- [x] Hook/policy after-turn runtime events
- [x] Hook/policy tool preflight guidance
- [x] Hook/policy tool blocking
- [x] Hook/policy after-tool guidance
- [x] Hook/policy budget override loading
- [x] Hook/policy safe-environment overlay for shell tools
- [x] Local manifest-backed MCP resource discovery
- [x] Local MCP resource listing and reading
- [x] MCP-backed runtime tools for local resource access
- [x] Real stdio MCP client transport for `initialize`, `resources/list`, `resources/read`, `tools/list`, and `tools/call`
- [x] Transport-backed MCP tool listing and execution
- [x] Local manifest-backed remote runtime discovery
- [x] Local remote profile listing and summary reporting
- [x] Local remote connect/disconnect state persistence
- [x] Local manifest/env-backed search runtime discovery
- [x] Local search-provider activation persistence
- [x] Provider-backed web search execution against configured search backends
- [x] Local persistent task runtime discovery
- [x] Local task create/get/list/update runtime flows
- [x] Local todo-list replacement runtime flow
- [x] Local persistent plan runtime discovery
- [x] Local plan get/update/clear runtime flows
- [x] Local plan-to-task sync flow
- [x] Dependency-aware local task state with blocking and actionable-task selection
- [x] Local task start/complete/block/cancel execution flows
- [x] Compaction metadata with compacted message ids
- [x] Compaction metadata with preserved-tail ids and compaction depth
- [x] Compaction metadata with compacted/preserved lineage ids and revision summaries
- [x] Compaction metadata with source mutation serials and mutation totals
- [x] Snipped-message metadata with source role/kind lineage
- [x] Snipped-message metadata with source lineage id and revision
- [x] Resume-time compaction / snipping replay reminder
- [x] Resume-time compaction replay of source mutation summaries
- [x] Query-engine facade that can drive the real Python runtime agent
- [x] Query-engine runtime event counters and transcript-kind summaries
- [x] Query-engine runtime mutation counters
- [x] Query-engine stream-level runtime summary event
- [x] Query-engine transcript-store compaction summaries
- [x] Delegate-group and delegated-subtask runtime events
- [x] Delegate-batch runtime events and summaries
- [x] Query-engine runtime orchestration summaries for group status and child stop reasons
- [x] Query-engine runtime context-reduction summaries
- [x] Query-engine runtime lineage summaries
- [x] Query-engine runtime resumed-child orchestration summaries

Missing:

- [ ] Full partial tool-result streaming parity across the complete upstream/npm tool surface
- [ ] Full rich transcript mutation behavior like the npm runtime beyond the current lineage, counters, block export, and mutation-serial tracking
- [ ] Full reasoning budgets and task budgets parity beyond the current cumulative model/tool/delegation/session-call enforcement
- [ ] Full multi-agent orchestration parity beyond dependency-aware batched delegation, resumed-child flows, and current agent-manager summaries
- [ ] Full file history snapshots and replay flows beyond the current preview/id-based implementation and delegated-batch replay metadata
- [ ] Full executable plugin lifecycle beyond manifest-driven prompt/tool/session hooks, blocking, aliases, virtual tools, and persisted runtime state
- [ ] Full session compaction / snipping parity beyond lineage-aware summaries, mutation-serial compaction metadata, and replay reminders
- [ ] Full `QueryEngine.ts` parity

## 2. CLI Entrypoints And Runtime Modes

Done:

- [x] Python CLI entrypoint
- [x] `agent` command
- [x] `agent-chat` command
- [x] `agent-resume` command
- [x] `agent-prompt` command
- [x] `agent-context` command
- [x] `agent-context-raw` command
- [x] Local background session mode
- [x] Local background session listing (`agent-ps`)
- [x] Local background session logs (`agent-logs`)
- [x] Local background attach snapshot (`agent-attach`)
- [x] Local background kill flow (`agent-kill`)
- [x] Local daemon-style background command family (`daemon start/ps/logs/attach/kill`)
- [x] Local daemon worker command path (`daemon worker`)
- [x] Local remote runtime CLI modes (`remote-mode`, `ssh-mode`, `teleport-mode`, `direct-connect-mode`, `deep-link-mode`)
- [x] Local remote runtime inspection commands (`remote-status`, `remote-profiles`, `remote-disconnect`)
- [x] Local account runtime inspection commands (`account-status`, `account-profiles`, `account-login`, `account-logout`)
- [x] Local search runtime inspection commands (`search-status`, `search-providers`, `search-activate`, `search`)
- [x] Local MCP runtime inspection commands (`mcp-status`, `mcp-resources`, `mcp-resource`, `mcp-tools`, `mcp-call-tool`)
- [x] Inventory/helper commands such as `summary`, `manifest`, `commands`, and `tools`

Missing:

- [ ] Full daemon supervisor parity beyond the current local daemon wrapper and worker flow
- [ ] Remote-control / bridge runtime mode
- [ ] Browser/native-host runtime mode
- [ ] Computer-use MCP mode
- [ ] Template job mode
- [ ] Environment runner mode
- [ ] Self-hosted runner mode
- [ ] tmux fast paths
- [ ] Worktree fast paths at the CLI entrypoint level
- [ ] Full `entrypoints/cli.tsx` and `entrypoints/init.ts` parity

## 3. Prompt Assembly

Done:

- [x] Structured Python system prompt builder
- [x] Intro/system/task/tool/tone/output sections
- [x] Session-specific prompt guidance
- [x] Environment-aware prompt sections
- [x] User context reminder injection
- [x] Custom system prompt override and append support
- [x] Local hook/policy guidance section in the Python system prompt
- [x] Local MCP guidance section in the Python system prompt
- [x] MCP transport/tool guidance section in the Python system prompt
- [x] Local remote-runtime guidance section in the Python system prompt
- [x] Local search-runtime guidance section in the Python system prompt
- [x] Local account-runtime guidance section in the Python system prompt
- [x] Local planning guidance section in the Python system prompt
- [x] Local task guidance section in the Python system prompt

Missing:

- [ ] Full parity with `constants/prompts.ts`
- [ ] Hook instruction sections
- [ ] MCP instruction sections
- [ ] Model-family-specific prompt variations
- [ ] Output-style variants
- [ ] Language-control sections
- [ ] Scratchpad prompt instructions
- [ ] More exact autonomous/proactive behavior sections
- [ ] Growthbook / feature-gated prompt sections
- [ ] Cyber / risk sections used upstream

## 4. Context Building And Memory

Done:

- [x] Current working directory snapshot
- [x] Shell / platform / date capture
- [x] Git status snapshot
- [x] `CLAUDE.md` discovery
- [x] Extra directory injection through `--add-dir`
- [x] Session context usage report
- [x] Tokenizer-aware context accounting with cached model-specific backends and heuristic fallback
- [x] Raw context inspection command
- [x] Plugin cache snapshot injection
- [x] Manifest-based plugin runtime summary injection
- [x] Manifest-based hook/policy summary injection
- [x] Trust-mode, managed-settings, and safe-env context injection
- [x] Manifest-based MCP runtime summary injection
- [x] Manifest-based MCP transport server summary injection
- [x] Manifest-based remote runtime summary injection
- [x] Manifest/env-based search runtime summary injection
- [x] Manifest-based account runtime summary injection
- [x] Manifest-based plan runtime summary injection
- [x] Manifest-based task runtime summary injection

Missing:

- [ ] Full tokenizer/chat-message framing parity beyond the current model-aware text token counters
- [ ] Full parity with `utils/queryContext.ts`
- [ ] Rich memory prompt loading
- [ ] Internal permission-aware memory handling
- [ ] Resume-aware prompt cache shaping used upstream
- [ ] More exact context cache invalidation rules
- [ ] Session context analysis parity
- [ ] Full memory subsystem parity

## 5. Slash Commands

Done:

- [x] `/help`
- [x] `/commands`
- [x] `/context`
- [x] `/usage`
- [x] `/context-raw`
- [x] `/env`
- [x] `/mcp`
- [x] `/mcp tools`
- [x] `/mcp tool <name>`
- [x] `/search`
- [x] `/remote`
- [x] `/remotes`
- [x] `/ssh`
- [x] `/teleport`
- [x] `/direct-connect`
- [x] `/deep-link`
- [x] `/disconnect`
- [x] `/account`
- [x] `/login`
- [x] `/logout`
- [x] `/resources`
- [x] `/resource`
- [x] `/plan`
- [x] `/planner`
- [x] `/tasks`
- [x] `/todo`
- [x] `/task`
- [x] `/task-next`
- [x] `/prompt`
- [x] `/system-prompt`
- [x] `/permissions`
- [x] `/hooks`
- [x] `/policy`
- [x] `/trust`
- [x] `/model`
- [x] `/tools`
- [x] `/memory`
- [x] `/status`
- [x] `/session`
- [x] `/clear`
- [x] `/config`
- [x] `/settings`

Missing:

- [ ] Full npm slash-command surface
- [x] Slash commands backed by MCP integration
- [ ] Slash commands tied to task/plan systems beyond the current local `/plan`, `/tasks`, and `/task` flows
- [ ] Slash commands tied to remote/background sessions beyond the current local remote connect/disconnect and background inspection flows
- [ ] Slash commands with richer interactive behavior
- [ ] Slash commands tied to plugins and bundled skills
- [ ] Slash commands tied to account, settings, and auth flows beyond the current local `/account`, `/login`, `/logout`, `/config`, and `/settings` flows

## 6. Built-in Tools

Done:

- [x] `list_dir`
- [x] `read_file`
- [x] `write_file`
- [x] `edit_file`
- [x] `glob_search`
- [x] `grep_search`
- [x] `bash`
- [x] `web_fetch`
- [x] `search_status`
- [x] `search_list_providers`
- [x] `search_activate_provider`
- [x] `web_search`
- [x] `tool_search`
- [x] `sleep`
- [x] `ask_user_question`
- [x] `account_status`
- [x] `account_list_profiles`
- [x] `account_login`
- [x] `account_logout`
- [x] `notebook_edit`
- [x] `mcp_list_resources`
- [x] `mcp_read_resource`
- [x] `mcp_list_tools`
- [x] `mcp_call_tool`
- [x] `remote_status`
- [x] `remote_list_profiles`
- [x] `remote_connect`
- [x] `remote_disconnect`
- [x] `config_list`
- [x] `config_get`
- [x] `config_set`
- [x] `plan_get`
- [x] `update_plan`
- [x] `plan_clear`
- [x] `delegate_agent`
- [x] `task_next`
- [x] `task_list`
- [x] `task_get`
- [x] `task_create`
- [x] `task_update`
- [x] `task_start`
- [x] `task_complete`
- [x] `task_block`
- [x] `task_cancel`
- [x] `todo_write`
- [x] `team_list`
- [x] `team_get`
- [x] `team_create`
- [x] `team_delete`
- [x] `send_message`
- [x] `team_messages`

Missing:

- [ ] Agent spawning tool parity beyond the current `delegate_agent` runtime tool
- [ ] Skill tool
- [ ] Web fetch parity beyond the current local text-fetch implementation
- [ ] Web search parity beyond the current provider-backed implementation
- [ ] LSP tool
- [ ] Tool search parity beyond the current local registry search
- [ ] Config tool
- [ ] Terminal capture tool
- [ ] Browser tool
- [ ] Workflow tool
- [ ] Remote trigger tool
- [ ] Sleep / cron tools beyond the current local `sleep` tool
- [ ] PowerShell tool parity
- [ ] Worktree enter/exit tools
- [ ] Full `tools.ts` parity

## 7. Commands And Task Systems

Done:

- [x] Basic local command dispatch for the Python runtime
- [x] Inventory view of mirrored command names
- [x] Local persistent task runtime with create/get/list/update flows
- [x] Local todo-list replacement flow
- [x] Local persistent plan runtime with get/update/clear flows
- [x] Local plan-to-task sync flow
- [x] Local dependency-aware task execution flow with next-task selection and blocked/unblocked state
- [x] Local remote profile/runtime flow with persisted connect/disconnect state
- [x] Local background task management for agent worker sessions
- [x] Local ask-user runtime with queued answers, history, and slash/CLI inspection flows
- [x] Local team runtime with persisted teams, messages, and slash/CLI inspection flows

Missing:

- [ ] Real implementation of the larger upstream command tree
- [ ] Task orchestration system beyond the current local dependency-aware task runtime
- [ ] Planner / task execution parity beyond the current local plan persistence, sync, and next-task flow
- [ ] Team / collaboration command flows beyond the current local team runtime and message recording flows
- [ ] Command-specific session behaviors
- [ ] Full `src/commands/*` parity
- [ ] Full `src/tasks/*` parity

## 8. Permissions, Hooks, And Policy

Done:

- [x] Read-only default mode
- [x] Write-gated mode
- [x] Shell-gated mode
- [x] Unsafe mode for destructive shell actions
- [x] Local hook/policy manifest discovery
- [x] Hook before-prompt and after-turn runtime handling
- [x] Hook/policy tool preflight, deny, and after-tool handling
- [x] Policy budget override loading
- [x] Managed settings loading and reporting
- [x] Safe environment loading for shell tool context
- [x] Trust reporting and hook/policy slash commands
- [x] Permission-denial runtime events for policy/tool blocks

Missing:

- [ ] Tool-permission workflow parity
- [ ] Trust-gated initialization
- [ ] Hook-config management
- [ ] Full hooks and policy parity

## 9. MCP, Plugins, And Skills

Done:

- [x] Placeholder mirrored package layout for plugins, skills, services, and remote subsystems
- [x] Local manifest-backed MCP discovery
- [x] Local MCP resource listing and reading
- [x] MCP-backed runtime tools for local resource access
- [x] Real MCP client support over local stdio transport
- [x] MCP server integration for stdio child-process servers
- [x] MCP-backed tool listing and execution over transport

Missing:

- [ ] Full MCP-backed tool parity beyond the current stdio resource/tool list/read/call support
- [ ] Plugin discovery and loading
- [ ] Bundled plugin support
- [ ] Plugin lifecycle management
- [ ] Plugin update/cache behavior
- [ ] Skill discovery and execution parity
- [ ] Bundled skill support
- [ ] Full plugin and skill parity

## 10. Interactive UI / REPL / TUI

Done:

- [x] Non-interactive CLI execution
- [x] Basic interactive REPL-style agent chat loop
- [x] Transcript printing for debugging

Missing:

- [ ] Interactive REPL parity beyond the current basic `agent-chat` loop
- [ ] Ink/TUI component parity
- [ ] Screen system parity
- [ ] Keyboard interaction parity
- [ ] Interactive status panes
- [ ] Approval UI flows
- [ ] Rich incremental rendering
- [ ] Full `components`, `screens`, and `ink` parity

## 11. Remote, Background, And Team Features

Done:

- [x] Session save/resume on local disk
- [x] Local manifest-backed remote profile/runtime state
- [x] Local remote connect/disconnect session state
- [x] Local background agent processes
- [x] Local background attach/log/kill workflows
- [x] Local daemon-style wrapper over background agent sessions

Missing:

- [ ] Real remote execution modes beyond the current local manifest-backed remote runtime and CLI/profile flows
- [ ] Team runtime features
- [ ] Team messaging features
- [ ] Shared remote state
- [ ] Upstream proxy runtime integration
- [ ] Full `remote`, `server`, `bridge`, `upstreamproxy`, and team parity

## 12. Editor, Platform, And Native Integrations

Done:

- [x] Standard shell-based local workflow

Missing:

- [ ] Voice mode parity
- [ ] VIM mode parity
- [ ] Keybinding parity
- [ ] Notification hooks
- [ ] Native TypeScript / platform helper parity
- [ ] JetBrains/editor integration parity
- [ ] Browser/native host integrations
- [ ] Platform-specific startup/shutdown logic

## 13. Services And Internal Subsystems

Done:

- [x] Minimal internal service layer required by the current Python runtime
- [x] Local account/auth runtime for manifest-backed profile discovery and persisted login state

Missing:

- [ ] Real service implementations for the mirrored `services` package
- [ ] Config service parity
- [ ] Account/auth service parity
- [ ] Analytics/telemetry service parity
- [ ] Growthbook/feature-flag parity
- [ ] GitHub / git helper parity
- [ ] Sandbox/settings utility parity
- [ ] Todo/task utility parity
- [ ] Internal helpers used by the upstream runtime

## 14. Mirrored Workspace Versus Working Runtime

Working Python runtime today:

- [x] `src/main.py`
- [x] `src/agent_runtime.py`
- [x] `src/agent_tools.py`
- [x] `src/agent_prompting.py`
- [x] `src/agent_context.py`
- [x] `src/agent_context_usage.py`
- [x] `src/agent_session.py`
- [x] `src/agent_slash_commands.py`
- [x] `src/account_runtime.py`
- [x] `src/config_runtime.py`
- [x] `src/agent_types.py`
- [x] `src/mcp_runtime.py`
- [x] `src/plan_runtime.py`
- [x] `src/plugin_runtime.py`
- [x] `src/remote_runtime.py`
- [x] `src/search_runtime.py`
- [x] `src/hook_policy.py`
- [x] `src/background_runtime.py`
- [x] `src/task.py`
- [x] `src/task_runtime.py`
- [x] `src/tokenizer_runtime.py`
- [x] `src/openai_compat.py`
- [x] `src/session_store.py`
- [x] `src/permissions.py`

Mirrored inventory / scaffold areas that still need real implementation work:

- [ ] `src/commands.py`
- [ ] `src/tools.py`
- [ ] `src/query_engine.py`
- [ ] `src/runtime.py`
- [ ] Remaining mirrored inventory surfaces still represented mainly by snapshot data under `src/reference_data/*`
- [ ] Command/task/plugin/skill/service/editor subsystems that exist upstream but do not yet have real Python modules after the tree cleanup

## 15. High-Priority Next Steps

- [ ] Expand the real Python tool registry toward upstream `tools.ts`
- [ ] Replace more snapshot-backed mirrored modules with working runtime code
- [ ] Expand MCP parity beyond the current stdio resource/tool transport support
- [ ] Expand hooks and policy parity beyond the current manifest/runtime implementation
- [ ] Build a real interactive REPL / TUI
- [ ] Expand background session parity beyond the current local worker/log/attach model
- [ ] Add real remote session transport and shared remote state beyond the current local remote-profile runtime
- [ ] Port more of the command/task system
- [ ] Close the gap between the mirrored workspace and the working runtime
