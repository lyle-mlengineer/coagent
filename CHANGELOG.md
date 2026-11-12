# Changelog

All notable changes to coAgent. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `src/` package layout with the module tree from `DESIGN.md` §11.
- `.coagent/config.toml` template shipped as package data, covering model,
  verify, git, durability, agent budgets, tools, artifacts, safety, and UI.
- `coagent.config` — TOML loading, per-section dataclasses, unknown-key
  rejection, hard 12,288-token context cap, and schema-version refusal
  (PRD FR-DIS-5).
- `coagent` CLI with `init`, `doctor`, `config`, `tools`, and `resume` verbs.
- `coagent doctor` reports the GPU envelope and whether it is a free-tier T4.
- Dev extras (`[llm]`, `[ui]`, `[context]`, `[colab]`, `[dev]`) so the package
  installs and tests on CPU without torch.

### Safety layer (`coagent.safety`)

- `redact` — GitHub, Hugging Face, OpenAI, Anthropic, AWS, Google, Stripe,
  Slack, JWT, URL-credential, `Bearer`, assigned-secret, and private-key
  redaction. Keeps the last four characters; scrubs nested dicts and lists.
  (PRD FR-PRV-2, FR-BOOT-5)
- `denylist` — tokenized destructive-command rules: force-push, push to `main`,
  `reset --hard`, `branch -D`, `rm -rf /`, `curl | sh`, fork bombs,
  shutdown/reboot, mkfs. Deny rules evaluate before allow rules (FR-GIT-9).
- `policy` — the approval matrix enforced in code. Reads are jailed (including
  `.git`); writes are allowed only inside the repo; shell/install/network/
  delete ask in interactive mode and **deny** in non-interactive mode
  (FR-TOL-4). `check_push` allows only configured remotes and `coagent/*`
  refs (FR-GIT-8).

### Tool layer (`coagent.tools`)

- Registry with JSON schemas, unknown-tool and unknown-argument rejection, and
  type coercion, all with corrective messages (FR-TOL-1).
- Per-phase tool sets: SURVEY is read-only, PLAN gets **no tools at all**,
  VERIFY cannot write (DESIGN §7.1).
- `read_file`, `list_dir`, `grep` — jailed reads, line-numbered output, and
  head/tail truncation (FR-CTX-3).
- `edit_file` — SEARCH/REPLACE with a unique-match requirement, a fuzzy
  nearest-region hint on failure, read-before-write enforcement (FR-AGT-4), and
  a syntax gate that rejects an unparseable result (FR-VER-4).
- `write_file` — new files, or full rewrites only under 200 lines.
- `run_shell`, `run_python` — process-group isolation with timeout kill, a
  scrubbed environment (no tokens reach the child), `CUDA_VISIBLE_DEVICES=""`
  by default (PRD D4), and a 64 KB head/tail output cap (FR-TOL-3).

### Git layer (`coagent.gitx`) — git as the agent's database

- `git.py` — a thin, jailed git runner: scrubbed env, `GIT_TERMINAL_PROMPT=0`
  (a missing credential errors instead of hanging), porcelain/-z output, and the
  push allowlist enforced *before* spawn. `GIT_ASKPASS` carries the token; it is
  never written to `.git/config` (DESIGN §9.2).
- `commit.py` — the six `Coagent-*` trailers from PRD Appendix B, values
  flattened to one line so a newline in a task cannot break `--grep`. Refuses
  an empty commit, which would corrupt step numbering.
- `session.py` — session branch `coagent/<date>-<id>`; **refuses a dirty tree**
  with the list of paths and a remedy (FR-GIT-1, AT-3); refuses to work on
  `main`; reuses an existing branch as a *resume*. coAgent's own runtime state
  (`.coagent/lock.json`, `tmp/`, `sessions/`) is excluded from the dirty check.
- `lock.py` — single-writer lock with heartbeat TTL, stale expiry, and `--force`
  takeover that reports the blocking session (FR-GIT-6).
- `durable.py` — `checkpoint()`: push the session branch, or fall back to a
  `git bundle` (full first, then `A..B` increments) when the push fails or a
  token is missing. A failed sync warns and continues (FR-GIT-4).
- `resume.py` — a template-extracted briefing (no LLM call) from branch
  trailers, the plan file, and the JSONL log; falls back to trailers when a
  hard kill left no log. `next_step` is monotonic across plan commits.
- `log.py` — JSONL session log, **redacted before the write** (FR-PRV-3).
- `coagent init` now also adds the `.coagent` runtime entries to `.gitignore`;
  the session path only warns rather than editing a tracked file.

### Fixed

- Pinning `core.autocrlf` in the git child env made every CRLF worktree look
  permanently modified. The value is now inherited, never overridden.
- `os.getpgid`/`start_new_session` do not exist on Windows; both are guarded so
  the tool suite runs on a dev laptop while Linux keeps real process-group kills.

### Tests

159 unit tests, CPU-only — no GPU and no network required.

## [0.1.0] - TBD

First release target: milestone M1. Fix a failing test in a toy repo end-to-end
on a free T4 (acceptance test AT-1).
