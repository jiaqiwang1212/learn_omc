<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# .claude

## Purpose
Claude Code configuration directory for this project. Contains the OMC CLAUDE.md (injected into every session), project settings, local permission overrides, and project-local skills.

## Key Files
| File | Description |
|------|-------------|
| `CLAUDE.md` | OMC orchestration instructions loaded into every Claude Code session in this project |
| `settings.json` | Enables the `oh-my-claudecode@omc` plugin for this project |
| `settings.local.json` | Local permission allowlist for setup automation (bash, npm, claude mcp, gh auth) |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `skills/` | Project-local skills available only within this workspace (see `skills/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Do **not** manually edit `CLAUDE.md` — it is managed by the OMC plugin; run `/oh-my-claudecode:omc-setup --local` to refresh it
- `settings.local.json` is gitignored by convention and holds developer-specific permissions
- New skills added under `skills/` are automatically available via the Skill tool in this project

### Testing Requirements
- After modifying `settings.json`, restart Claude Code for changes to take effect
- Verify skill loading with `/reload-plugins`

### Common Patterns
- Skills follow the format: `skills/<name>/SKILL.md` with YAML frontmatter (`name`, `description`, `user-invocable`)

## Dependencies

### Internal
- `skills/omc-reference/` — OMC agent catalog and routing reference

### External
- Claude Code plugin system — reads `settings.json` to enable plugins

<!-- MANUAL: -->
