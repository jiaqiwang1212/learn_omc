<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# omc-reference

## Purpose
Built-in OMC reference skill providing the agent catalog, available tools, team pipeline routing, commit protocol, and skills registry. Auto-loads when delegating to agents, using OMC tools, orchestrating teams, making commits, or invoking skills. Not user-invocable — it is an internal context source.

## Key Files
| File | Description |
|------|-------------|
| `SKILL.md` | Skill definition with frontmatter and full OMC agent/tool reference content |

## For AI Agents

### Working In This Directory
- This skill is managed by the OMC plugin and installed automatically by `/oh-my-claudecode:omc-setup`
- Do not manually edit `SKILL.md` — it will be overwritten on the next OMC update
- To refresh this skill, run `/oh-my-claudecode:omc-setup --local`

### Common Patterns
- Frontmatter: `user-invocable: false` — this is a passive reference, not a slash command
- Consumed automatically when Claude Code needs agent routing decisions

## Dependencies

### External
- `oh-my-claudecode` plugin — installs and manages this skill

<!-- MANUAL: -->
