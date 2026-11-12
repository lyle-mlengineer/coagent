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
- `coagent` CLI with `init`, `doctor`, `config`, and `tools` verbs.
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

### Tests

124 unit tests, CPU-only — no GPU and no network required.

## [0.1.0] - TBD

First release target: milestone M1. Fix a failing test in a toy repo end-to-end
on a free T4 (acceptance test AT-1).
