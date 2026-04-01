<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# learn_omc

## Purpose
A learning workspace for exploring and configuring oh-my-claudecode (OMC), the multi-agent orchestration layer for Claude Code. This project exists to experiment with OMC setup, skills, agents, and MCP server integrations.

## Key Files
| File | Description |
|------|-------------|
| `.claude/CLAUDE.md` | OMC configuration injected into every Claude Code session |
| `.claude/settings.json` | Project-level plugin and enablement settings |
| `.claude/settings.local.json` | Local permission overrides for setup automation |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `.claude/` | Claude Code configuration — skills, settings, and OMC CLAUDE.md (see `.claude/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- This is a configuration-only workspace with no application source code
- The primary artifact is `.claude/CLAUDE.md`, which is managed by the OMC plugin and should not be manually edited between OMC versions
- Use `/oh-my-claudecode:omc-setup` to refresh or reconfigure OMC

### Testing Requirements
- Verify OMC is working by running `claude mcp list` and confirming connected servers
- Check `~/.claude/.omc-config.json` for persisted preferences

### Common Patterns
- All OMC configuration lives under `.claude/`
- MCP server configs are stored in `~/.claude.json` (project scope) or `~/.claude/claude.json` (global)
- Skills are loaded from `.claude/skills/`

## Dependencies

### External
- `oh-my-claudecode` (plugin) — OMC orchestration layer
- `context7` MCP — library documentation
- `filesystem` MCP — extended file access at `~/`

<!-- MANUAL: -->
