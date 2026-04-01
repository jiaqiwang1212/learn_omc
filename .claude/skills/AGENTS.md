<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-01 | Updated: 2026-04-01 -->

# skills

## Purpose
Project-local skills directory. Skills here are available only within this workspace and are loaded by the Claude Code plugin system on startup. Each skill is a subdirectory containing a `SKILL.md` with YAML frontmatter and the skill's instruction body.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `omc-reference/` | OMC agent catalog, tool routing, and commit protocol reference (see `omc-reference/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Each skill directory must contain a `SKILL.md` file with valid YAML frontmatter (`name`, `description`, `user-invocable`)
- Skills with `user-invocable: false` are internal references loaded automatically, not invocable via `/skill-name`
- After adding or modifying a skill, run `/reload-plugins` to apply changes

### Testing Requirements
- Verify a skill loads by checking it appears in the skill list after `/reload-plugins`
- For user-invocable skills, test with `/oh-my-claudecode:<skill-name>`

### Common Patterns
```
skills/
└── <skill-name>/
    └── SKILL.md    ← frontmatter + instructions
```

<!-- MANUAL: -->
