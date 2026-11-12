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
- `coagent` CLI with `init`, `doctor`, `config`, `tools`, `resume`, and
  `models` verbs.
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

### Agent loop (`coagent.agent`)

- `prompts.py` — the five phase prompts in one auditable file. Tool results are
  wrapped in `<observation>` and the system prompt declares them untrusted data,
  which is the first layer of the prompt-injection defence (DESIGN §9.3).
  PLAN's prompt states it has no tools.
- `budget.py` — hard ceilings on steps, wall clock, tokens, GPU minutes,
  commits, and consecutive failures. Exhaustion raises `BudgetExhausted` so the
  run checkpoints and reports instead of burning the session (FR-AGT-5).
  `LoopDetector` fingerprints `(tool, args)`; 3 repeats force a replan, 5 stop
  the run (DESIGN §7.4).
- `state.py` — `PlanFile`, the durable plan in `.coagent/plans/`, written before
  the first edit and re-read after eviction, so memory survives both context
  loss and VM death (FR-CTX-5). Observations are a bounded window.
- `loop.py` — the explicit state machine SURVEY → PLAN → ACT → VERIFY →
  CHECKPOINT → REPORT. PLAN structurally cannot act; one write or exec per
  turn; read-before-write; verify-then-commit; `done` is a predicate over
  passing checks and plan completion, never the model's opinion (FR-AGT-8).
- `coagent run "<task>"` with `--auto` and `--dry-run` (scripted model, no GPU).
- `python -m coagent` for use inside a Colab cell.

### Fixed

- `git commit` failed with a bare exit 1 and empty stderr because nothing was
  staged. The loop now stages changed and new files, never `.coagent/` state.
- Plan steps were never ticked, so no task could ever satisfy the definition of
  done. A successful edit ticks the step naming that file, and passing checks
  tick the step naming them.
- `Budget.check()` used `>=`, so the Nth step raised instead of the (N+1)th.
  Ceilings are now inclusive.
- A final checkpoint committed failing work labelled as verified. The trailer
  now records `failed (work in progress)`.
- The report phase could echo a raw JSON tool envelope at the user. It is now
  detected with the real parser and replaced with a generated summary.
- A failed commit is no longer silent: the reason is emitted as a `commit_failed`
  event.

### Model layer (`coagent.llm`)

- `base.py` — the `LLMBackend` interface (`generate`, `count_tokens`, streaming,
  cancel, `load`/`unload`), plus `TokenBudget`, which rejects an over-cap request
  **before** generation using the model's own tokenizer (PRD D2, FR-CTX-6), and
  `FakeBackend`, a scripted backend that makes the whole loop testable on CPU.
- `grammar.py` — tolerant envelope extraction with a 5-step escalation ladder:
  labelled fence → bare fence → bare `{...}` (balanced-brace scan that respects
  strings) → JSON5-ish repair (single quotes, unquoted keys, trailing commas,
  literal control characters inside strings) → failure with a **corrective
  message** the loop feeds straight back. Never raises `JSONDecodeError` at the
  caller. Failures are diagnosed individually: "cut off" (unbalanced braces,
  with advice to send a smaller edit), "malformed" (bad escapes, missing comma),
  and "no JSON object" (plain prose) are different problems and get different
  advice.
- `registry.py` — exactly two validated configurations (7B/NF4/12k and
  3B/NF4/8k fallback), and `preflight_gpu()`, which checks T4 / compute 7.5 /
  ≥14 GB VRAM and returns an actionable refusal instead of raising (PRD D1).
- `transformers_backend.py` — NF4 4-bit with fp16 compute and
  `attn_implementation="sdpa"` (Turing has no bf16 and no FlashAttention-2),
  `expandable_segments` set before torch import, `TextIteratorStreamer` on a
  worker thread, a real `StoppingCriteria` for cancel, and the OOM ladder
  (`shrink_context` → `halve_output` → `fallback_model` → `stop`) that unloads
  the model and never loses the session. Torch and transformers are resolved
  through module-level accessors, so `import coagent` stays torch-free and the
  orchestration is testable without a GPU.

### Fixed

- `_is_cuda_oom` missed a wrapped `OutOfMemoryError` that only appeared in the
  message text; the OOM ladder now recognises it.
- `OutOfMemoryLadder.exhausted` reported False after all three recovery actions;
  it now flips once the ladder is spent.

- `.coagent/plans/` was missing from the runtime paths, so a second run refused
  to start on coAgent's own plan files.

### Tests

256 unit and integration tests, CPU-only — no GPU and no network required.
Includes end-to-end runs on a real git repository: a scripted model drives
SURVEY → PLAN → ACT → VERIFY → commit with trailers, and separate tests prove
the dirty-tree refusal, plan rejection before any write, honest verify trailers,
and clean budget exhaustion.

## [0.1.0] - TBD

First release target: milestone M1. Fix a failing test in a toy repo end-to-end
on a free T4 (acceptance test AT-1).
