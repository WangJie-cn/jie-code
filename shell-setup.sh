#!/usr/bin/env bash
# jie-code shell integration
# Add to your .bashrc or .zshrc:
#   source /data/jie-code/shell-setup.sh

export JIE_CONFIG_HOME="${JIE_CONFIG_HOME:-$HOME/.jie}"

# Aliases
alias jc='jie-code'

# Quick profile shortcuts
alias jc-claude='jie-code agent --profile anthropic-proxy'
alias jc-opus='jie-code agent --profile anthropic-opus'
alias jc-local='jie-code agent --profile local-qwen'
alias jc-ollama='jie-code agent --profile ollama'

# Completion (basic)
_jc_completions() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local commands="agent agent-chat agent-resume summary manifest commands tools config-status mcp-status"
    local profiles="anthropic-proxy anthropic-opus local-qwen ollama openrouter"

    if [ "${COMP_WORDS[COMP_CWORD-1]}" = "--profile" ]; then
        COMPREPLY=($(compgen -W "$profiles" -- "$cur"))
    elif [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=($(compgen -W "$commands" -- "$cur"))
    fi
}
complete -F _jc_completions jie-code
complete -F _jc_completions jc
