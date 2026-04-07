from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .agent_runtime import LocalCodingAgent


@dataclass(frozen=True)
class ParsedSlashCommand:
    command_name: str
    args: str
    is_mcp: bool


@dataclass(frozen=True)
class SlashCommandResult:
    handled: bool
    should_query: bool
    prompt: str | None = None
    output: str = ''
    transcript: tuple[dict[str, Any], ...] = ()


SlashCommandHandler = Callable[['LocalCodingAgent', str, str], SlashCommandResult]


@dataclass(frozen=True)
class SlashCommandSpec:
    names: tuple[str, ...]
    description: str
    handler: SlashCommandHandler


def parse_slash_command(input_text: str) -> ParsedSlashCommand | None:
    trimmed = input_text.strip()
    if not trimmed.startswith('/'):
        return None

    without_slash = trimmed[1:]
    words = without_slash.split(' ')
    if not words or not words[0]:
        return None

    command_name = words[0]
    is_mcp = False
    args_start_index = 1
    if len(words) > 1 and words[1] == '(MCP)':
        command_name = f'{command_name} (MCP)'
        is_mcp = True
        args_start_index = 2

    return ParsedSlashCommand(
        command_name=command_name,
        args=' '.join(words[args_start_index:]),
        is_mcp=is_mcp,
    )


def looks_like_command(command_name: str) -> bool:
    return re.search(r'[^a-zA-Z0-9:\-_]', command_name) is None


def preprocess_slash_command(
    agent: 'LocalCodingAgent',
    input_text: str,
) -> SlashCommandResult:
    if not input_text.strip().startswith('/'):
        return SlashCommandResult(handled=False, should_query=True, prompt=input_text)

    parsed = parse_slash_command(input_text)
    if parsed is None:
        return _local_result(
            input_text,
            'Commands are in the form `/command [args]`.',
        )

    normalized_name = (
        parsed.command_name[:-6]
        if parsed.is_mcp and parsed.command_name.endswith(' (MCP)')
        else parsed.command_name
    )
    spec = find_slash_command(normalized_name)
    if spec is None:
        if looks_like_command(parsed.command_name):
            label = normalized_name if parsed.is_mcp else parsed.command_name
            return _local_result(input_text, f'Unknown skill: {label}')
        return SlashCommandResult(handled=False, should_query=True, prompt=input_text)

    return spec.handler(agent, parsed.args.strip(), input_text)


def get_slash_command_specs() -> tuple[SlashCommandSpec, ...]:
    return (
        SlashCommandSpec(
            names=('help', 'commands'),
            description='Show the built-in Python slash commands.',
            handler=_handle_help,
        ),
        SlashCommandSpec(
            names=('context', 'usage'),
            description='Show estimated session context usage similar to the npm /context command.',
            handler=_handle_context,
        ),
        SlashCommandSpec(
            names=('context-raw', 'env'),
            description='Show the raw environment, user context, and system context snapshot.',
            handler=_handle_context_raw,
        ),
        SlashCommandSpec(
            names=('mcp',),
            description='Show discovered local MCP manifests and resource counts.',
            handler=_handle_mcp,
        ),
        SlashCommandSpec(
            names=('search',),
            description='Show search runtime status, list or activate providers, or run a real web search query.',
            handler=_handle_search,
        ),
        SlashCommandSpec(
            names=('remote',),
            description='Show local remote runtime status or activate a remote target/profile.',
            handler=_handle_remote,
        ),
        SlashCommandSpec(
            names=('worktree',),
            description='Show managed git worktree status or enter/exit the current managed worktree session.',
            handler=_handle_worktree,
        ),
        SlashCommandSpec(
            names=('account',),
            description='Show local account runtime status or configured account profiles.',
            handler=_handle_account,
        ),
        SlashCommandSpec(
            names=('ask',),
            description='Show local ask-user runtime status or ask-user history.',
            handler=_handle_ask,
        ),
        SlashCommandSpec(
            names=('login',),
            description='Activate a local account profile or ephemeral identity.',
            handler=_handle_login,
        ),
        SlashCommandSpec(
            names=('logout',),
            description='Clear the active local account session.',
            handler=_handle_logout,
        ),
        SlashCommandSpec(
            names=('config', 'settings'),
            description='Show local config runtime state, effective config, config sources, or a config value.',
            handler=_handle_config,
        ),
        SlashCommandSpec(
            names=('remotes',),
            description='List configured local remote profiles.',
            handler=_handle_remotes,
        ),
        SlashCommandSpec(
            names=('ssh',),
            description='Activate a local SSH remote target/profile.',
            handler=_handle_ssh,
        ),
        SlashCommandSpec(
            names=('teleport',),
            description='Activate a local teleport remote target/profile.',
            handler=_handle_teleport,
        ),
        SlashCommandSpec(
            names=('direct-connect',),
            description='Activate a local direct-connect remote target/profile.',
            handler=_handle_direct_connect,
        ),
        SlashCommandSpec(
            names=('deep-link',),
            description='Activate a local deep-link remote target/profile.',
            handler=_handle_deep_link,
        ),
        SlashCommandSpec(
            names=('disconnect', 'remote-disconnect'),
            description='Disconnect the active local remote runtime target.',
            handler=_handle_remote_disconnect,
        ),
        SlashCommandSpec(
            names=('resources',),
            description='List local MCP resources, optionally filtered by a query string.',
            handler=_handle_resources,
        ),
        SlashCommandSpec(
            names=('resource',),
            description='Render a local MCP resource by URI.',
            handler=_handle_resource,
        ),
        SlashCommandSpec(
            names=('tasks', 'todo'),
            description='Show the local runtime task list, optionally filtered by status.',
            handler=_handle_tasks,
        ),
        SlashCommandSpec(
            names=('workflows',),
            description='List local workflows discovered from workflow manifests.',
            handler=_handle_workflows,
        ),
        SlashCommandSpec(
            names=('workflow',),
            description='Show or run one local workflow by name.',
            handler=_handle_workflow,
        ),
        SlashCommandSpec(
            names=('triggers',),
            description='List local remote triggers discovered from remote trigger manifests.',
            handler=_handle_triggers,
        ),
        SlashCommandSpec(
            names=('trigger',),
            description='Show or run one local remote trigger by id.',
            handler=_handle_trigger,
        ),
        SlashCommandSpec(
            names=('teams',),
            description='List the locally configured collaboration teams.',
            handler=_handle_teams,
        ),
        SlashCommandSpec(
            names=('team',),
            description='Show one local collaboration team by name.',
            handler=_handle_team,
        ),
        SlashCommandSpec(
            names=('messages',),
            description='Show recorded collaboration messages for all teams or one team.',
            handler=_handle_messages,
        ),
        SlashCommandSpec(
            names=('task-next', 'next-task'),
            description='Show the next actionable tasks from the local runtime task list.',
            handler=_handle_task_next,
        ),
        SlashCommandSpec(
            names=('plan', 'planner'),
            description='Show the current local runtime plan.',
            handler=_handle_plan,
        ),
        SlashCommandSpec(
            names=('task',),
            description='Show a local runtime task by id.',
            handler=_handle_task,
        ),
        SlashCommandSpec(
            names=('prompt', 'system-prompt'),
            description='Render the effective Python system prompt.',
            handler=_handle_prompt,
        ),
        SlashCommandSpec(
            names=('permissions',),
            description='Show the active tool permission mode.',
            handler=_handle_permissions,
        ),
        SlashCommandSpec(
            names=('hooks', 'policy'),
            description='Show discovered local hook and policy manifests.',
            handler=_handle_hooks,
        ),
        SlashCommandSpec(
            names=('trust',),
            description='Show workspace trust mode, managed settings, and safe environment values.',
            handler=_handle_trust,
        ),
        SlashCommandSpec(
            names=('model',),
            description='Show or update the active model for the current agent instance.',
            handler=_handle_model,
        ),
        SlashCommandSpec(
            names=('tools',),
            description='List the registered tools and whether the current permissions allow them.',
            handler=_handle_tools,
        ),
        SlashCommandSpec(
            names=('memory',),
            description='Show the currently loaded CLAUDE.md memory bundle and discovered files.',
            handler=_handle_memory,
        ),
        SlashCommandSpec(
            names=('status', 'session'),
            description='Show a short runtime/session status summary.',
            handler=_handle_status,
        ),
        SlashCommandSpec(
            names=('clear',),
            description='Clear ephemeral Python runtime state for this process.',
            handler=_handle_clear,
        ),
    )


def find_slash_command(command_name: str) -> SlashCommandSpec | None:
    lowered = command_name.lower()
    for spec in get_slash_command_specs():
        if lowered in spec.names:
            return spec
    return None


def _handle_help(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    lines = ['# Slash Commands', '']
    for spec in get_slash_command_specs():
        primary = f'/{spec.names[0]}'
        aliases = ', '.join(f'/{name}' for name in spec.names[1:])
        label = f'{primary} ({aliases})' if aliases else primary
        lines.append(f'- `{label}`: {spec.description}')
    lines.extend(
        [
            '',
            'These commands are handled locally before the model loop, similar to the npm runtime.',
        ]
    )
    return _local_result(input_text, '\n'.join(lines))


def _handle_context(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    prompt = args or None
    return _local_result(input_text, agent.render_context_report(prompt))


def _handle_context_raw(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_context_snapshot_report())


def _handle_mcp(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_mcp_report())
    if command == 'tools':
        return _local_result(input_text, agent.render_mcp_tools_report())
    if command.startswith('tools '):
        query = command.split(' ', 1)[1].strip()
        return _local_result(input_text, agent.render_mcp_tools_report(query or None))
    if command.startswith('tool '):
        tool_name = command.split(' ', 1)[1].strip()
        if not tool_name:
            return _local_result(input_text, 'Usage: /mcp tool <tool-name>')
        return _local_result(input_text, agent.render_mcp_call_tool_report(tool_name))
    return _local_result(input_text, agent.render_mcp_report(command))


def _handle_search(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_search_report())
    if command == 'providers':
        return _local_result(input_text, agent.render_search_providers_report())
    if command.startswith('providers '):
        query = command.split(' ', 1)[1].strip()
        return _local_result(input_text, agent.render_search_providers_report(query or None))
    if command.startswith('provider '):
        provider = command.split(' ', 1)[1].strip()
        if not provider:
            return _local_result(input_text, 'Usage: /search provider <name>')
        return _local_result(input_text, agent.render_search_report(provider=provider))
    if command.startswith('use '):
        provider = command.split(' ', 1)[1].strip()
        if not provider:
            return _local_result(input_text, 'Usage: /search use <name>')
        return _local_result(input_text, agent.render_search_activate_report(provider))
    return _local_result(input_text, agent.render_search_report(command))


def _handle_remote(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    target = args or None
    return _local_result(input_text, agent.render_remote_report(target))


def _handle_worktree(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_worktree_report())
    if command == 'history':
        return _local_result(input_text, agent.render_worktree_history_report())
    if command.startswith('enter'):
        name = command.split(' ', 1)[1].strip() if ' ' in command else None
        return _local_result(input_text, agent.render_worktree_enter_report(name or None))
    if command.startswith('exit'):
        parts = command.split()
        action = parts[1] if len(parts) > 1 else 'keep'
        discard_changes = any(part in {'discard', 'discard_changes=true'} for part in parts[2:])
        return _local_result(
            input_text,
            agent.render_worktree_exit_report(
                action=action,
                discard_changes=discard_changes,
            ),
        )
    return _local_result(
        input_text,
        'Usage: /worktree [history|enter <name>|exit <keep|remove> [discard]]',
    )


def _handle_account(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_account_report())
    if command == 'profiles':
        return _local_result(input_text, agent.render_account_profiles_report())
    if command.startswith('profile '):
        profile = command.split(' ', 1)[1].strip()
        if not profile:
            return _local_result(input_text, 'Usage: /account profile <name>')
        return _local_result(input_text, agent.render_account_report(profile))
    return _local_result(input_text, 'Usage: /account [profiles|profile <name>]')


def _handle_ask(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_ask_user_report())
    if command == 'history':
        return _local_result(input_text, agent.render_ask_user_history_report())
    return _local_result(input_text, 'Usage: /ask [history]')


def _handle_login(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    target = args.strip()
    if not target:
        return _local_result(input_text, 'Usage: /login <profile-or-identity>')
    return _local_result(input_text, agent.render_account_login_report(target))


def _handle_logout(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_account_logout_report())


def _handle_config(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_config_report())
    if command == 'effective':
        return _local_result(input_text, agent.render_config_effective_report())
    if command.startswith('source '):
        source = command.split(' ', 1)[1].strip()
        if not source:
            return _local_result(input_text, 'Usage: /config source <source-name>')
        return _local_result(input_text, agent.render_config_source_report(source))
    if command.startswith('get '):
        key_path = command.split(' ', 1)[1].strip()
        if not key_path:
            return _local_result(input_text, 'Usage: /config get <key-path>')
        return _local_result(input_text, agent.render_config_value_report(key_path))
    return _local_result(
        input_text,
        'Usage: /config [effective|source <name>|get <key-path>]',
    )


def _handle_remotes(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_remote_profiles_report(query))


def _handle_ssh(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /ssh <target-or-profile>')
    return _local_result(input_text, agent.render_remote_mode_report(args, mode='ssh'))


def _handle_teleport(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /teleport <target-or-profile>')
    return _local_result(input_text, agent.render_remote_mode_report(args, mode='teleport'))


def _handle_direct_connect(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /direct-connect <target-or-profile>')
    return _local_result(input_text, agent.render_remote_mode_report(args, mode='direct-connect'))


def _handle_deep_link(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /deep-link <target-or-profile>')
    return _local_result(input_text, agent.render_remote_mode_report(args, mode='deep-link'))


def _handle_remote_disconnect(
    agent: 'LocalCodingAgent',
    _args: str,
    input_text: str,
) -> SlashCommandResult:
    return _local_result(input_text, agent.render_remote_disconnect_report())


def _handle_resources(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_mcp_resources_report(query))


def _handle_resource(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /resource <mcp-resource-uri>')
    return _local_result(input_text, agent.render_mcp_resource_report(args))


def _handle_tasks(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    status = args or None
    return _local_result(input_text, agent.render_tasks_report(status))


def _handle_workflows(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_workflows_report(query))


def _handle_workflow(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, 'Usage: /workflow <name> | /workflow run <name>')
    if command.startswith('run '):
        workflow_name = command.split(' ', 1)[1].strip()
        if not workflow_name:
            return _local_result(input_text, 'Usage: /workflow run <name>')
        return _local_result(input_text, agent.render_workflow_run_report(workflow_name))
    return _local_result(input_text, agent.render_workflow_report(command))


def _handle_triggers(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_remote_triggers_report(query))


def _handle_trigger(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, 'Usage: /trigger <id> | /trigger run <id>')
    if command.startswith('run '):
        trigger_id = command.split(' ', 1)[1].strip()
        if not trigger_id:
            return _local_result(input_text, 'Usage: /trigger run <id>')
        return _local_result(
            input_text,
            agent.render_remote_trigger_action_report('run', trigger_id=trigger_id),
        )
    return _local_result(input_text, agent.render_remote_trigger_report(command))


def _handle_teams(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_teams_report(query))


def _handle_team(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    team_name = args.strip()
    if not team_name:
        return _local_result(input_text, 'Usage: /team <team-name>')
    return _local_result(input_text, agent.render_team_report(team_name))


def _handle_messages(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    team_name = args.strip() or None
    return _local_result(input_text, agent.render_team_messages_report(team_name))


def _handle_task_next(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_next_tasks_report())


def _handle_plan(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_plan_report())


def _handle_task(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /task <task-id>')
    return _local_result(input_text, agent.render_task_report(args))


def _handle_prompt(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_system_prompt())


def _handle_permissions(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_permissions_report())


def _handle_hooks(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_hook_policy_report())


def _handle_trust(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_trust_report())


def _handle_model(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, f'Current model: {agent.model_config.model}')
    agent.set_model(args)
    return _local_result(input_text, f'Set model to {agent.model_config.model}')


def _handle_tools(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_tools_report())


def _handle_memory(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_memory_report())


def _handle_status(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_status_report())


def _handle_clear(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    agent.clear_runtime_state()
    return _local_result(
        input_text,
        'Cleared ephemeral Python agent state for this process.',
    )


def _local_result(input_text: str, output: str) -> SlashCommandResult:
    transcript = (
        {'role': 'user', 'content': input_text},
        {'role': 'assistant', 'content': output},
    )
    return SlashCommandResult(
        handled=True,
        should_query=False,
        output=output,
        transcript=transcript,
    )
