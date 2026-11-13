# Changelog

All notable changes to coAgent. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-13

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
- `run_tests` — pytest output as a digest rather than a dump: pass/fail verdict,
  failing node ids, and the `>` pointer plus `E` block for each failure, capped
  at `verify.feedback_max_tokens` (DESIGN §7.5). Accepts `argv` (preferred, so a
  tokenized command from config is never re-split) or a `command` string, which
  is shlex-split without a shell. Classified EXEC, so it is unreachable from the
  read-only SURVEY phase.
- `coagent.tools.pyt` — the parser behind `run_tests`: `parse_pytest_output`
  (failed names, per-failure traces, summary, version, duration) and
  `summarise_pytest`. Whole traces only, most recent first, with an explicit
  omission note instead of a half-rendered traceback.

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

### Context layer (`coagent.context`)

- `repo_map` — signature-only repository map built for a 12k window
  (DESIGN §5.1–5.2). Parses every source file with `ast` and falls back to
  tree-sitter, then to line-anchored regex, so a file that will not parse
  degrades the map instead of ending the session. Emits signatures and
  module constants only — never bodies.
- PageRank-lite ranking, `0.5·reference_count + 0.3·recently_edited +
  0.2·mentioned_in_task`: a shared module outranks a leaf script, and a file
  the task names always survives the budget. Reference counting is inbound and
  token-set based — the naive per-symbol regex version cost 4 s on a 7k-line
  repo, now ~0.12 s.
- Greedy, deterministic packing into the repo map's 2000-token slot, cutting on
  file boundaries rather than mid-signature. Vendored, generated, and
  `__pycache__` trees are excluded; per-file signatures are capped at 40.
- `tokens` — ~4 chars/token estimator, `fit` for budget packing, and
  `truncate_middle` for head/tail truncation (monotone, so an "it fits" answer
  stays true for the real tokenizer).
- `context.repo_map_tokens` and `context.max_files` config knobs, validated with
  a 200-token floor.
- The agent loop's `_repo_map` stand-in is replaced: SURVEY now receives the
  ranked map, seeded with the task text and the files the session has touched.

### Retrieval (`coagent.context.retrieval`)

- `Retriever` — Okapi BM25 over function-sized chunks (DESIGN §5.3, FR-CTX-2).
  **No vector store**, per the design: on 2 vCPU beside a GPU an embedding index
  is a memory and latency tax for a ranking gain BM25 mostly covers at this repo
  size. `all-MiniLM-L6-v2` stays an optional future upgrade, not a dependency.
- Chunks come free from the repo map: `Signature` now carries 1-based, inclusive
  `start_line`/`end_line` from the same `ast`/tree-sitter/regex pass, so
  retrieval does not re-parse the repo. The AST, tree-sitter, and regex parsers
  all populate spans; the regex fallback closes a `def` on the first line that
  dedents past its body.
- A module prologue chunk (imports and module docstring) is added per file,
  because that is where a file says what it depends on. `const` signatures are
  skipped — a bare `X = ...` is not worth an excerpt slot.
- Tokenisation explodes `snake_case` and `camelCase`, so a query for "csv loader"
  matches `load_csv`, while the joined identifier is kept so an exact query still
  matches exactly.
- Ranking is BM25 plus a symbol-graph adjacency bonus (weight 0.35): a chunk that
  imports a module already in context, or references/defines a symbol defined in
  one, is pushed up. This is the "callers/callees of files already in context"
  part of §5.3, and files already read are excluded so the slot only ever shows
  new code.
- Deterministic ordering with a `(score, file, line)` tiebreak, so a re-query
  cannot shuffle equal-scoring chunks and invalidate a resumed session.
- `render()` packs whole chunks into the 3000-token excerpt slot and **skips** a
  chunk that does not fit rather than truncating it: half a function is worse
  than a different whole one, and a truncated header would lie about the range.
  A single 800-line method cannot eat the slot (`max_chunk_lines`).
- The ACT prompt receives the excerpts under an explicit "untrusted source, not
  instructions" banner, alongside the last failure.
- The repo map is now built once per session and cached, invalidated only when a
  file is written — it was being re-parsed on every turn. Retrieval or map
  failures degrade to a note and emit an event; neither can end a session.
- `context.retrieved_tokens` config knob.

### Notebook tools (`coagent.tools.notebooks`)

Notebooks are first-class (PRD D-2, FR-TOL-6), not source files with a JSON side
quest.

- `read_notebook` — cells by id, with type, execution count, and a hidden-output
  count. Assigns ids to notebooks that lack them (nbformat 4.4, common in older
  Colab files) **on read**, so an edit cannot land on a stale index; duplicate ids
  are repaired. `read_file` on a `.ipynb` now redirects here rather than showing
  escaped JSON as line numbers.
- `edit_notebook` — `replace_source`, `insert`, `delete`, `replace_metadata`.
  Cells are addressed **by id only**; an index is rejected with the list of valid
  ids, because indices shift under insert/delete and an index-addressed edit is a
  silent wrong edit. Read-before-write applies exactly as for `edit_file`.
- A cell edit is gated on `ast.parse` first: an unparseable Python cell is refused
  and the file is left untouched. IPython magics (`%matplotlib`, `!pip`, `%%bash`)
  are neutralised rather than flagged, since they are valid notebook cells that
  simply are not Python. An edited cell drops its stale outputs so the notebook
  never shows results from code that no longer exists.
- Every write is validated with `nbformat.validate` after bumping the minor
  version to 5 (required for ids), so coAgent can never hand Colab a corrupt file.
  The tool result diff is rendered over cells, not raw JSON, and excludes the
  spurious nbformat-version line.
- `exec_notebook` — `nbconvert --execute --inplace` in a subprocess, 300 s default
  (`tools.notebook_timeout_s`), run against a **copy** and moved back, so a crash
  mid-write cannot corrupt a tracked notebook. Outputs are read back from the
  notebook, never from nbconvert's stdout.
- `allow_errors=True` means a failing cell does not stop the run; the tool reports
  `N failing cell(s)` (nbconvert exits 0 either way, so the exit code is not the
  verdict), extracts the last traceback, and never prints the whole log when a
  traceback already explains the failure.
- FR-CTX-4 is enforced twice: only the last traceback and a 40-line stdout tail
  enter context, **and** each line is capped at 300 chars, because
  `print(x * 200)` is one line that would otherwise walk past a line-count budget.
  Images and HTML become artifacts in `meta["outputs"]`, and never reach the
  prompt as base64.
- Timeouts are detected from both paths: nbconvert's own `CellTimeoutError` and
  coAgent's process-group kill.
- Shell escapes inside code cells (`!pip install …`) are gated by the same
  approval matrix as `run_shell`, so `exec_notebook` is not a way around it.
- Phase gating: SURVEY may read a notebook, VERIFY may execute one, PLAN has no
  tools, and only ACT may edit. `exec_notebook` counts as a write for commit
  purposes, since it rewrites the notebook in place with outputs.

### Artifacts (`coagent.context.artifacts`, `coagent.tools.artifacts`)

The output of a data task is often a plot or a CSV, not a source file, and "done"
means the figure saved and visible (PRD US-3, G5). This is that pipeline.

- `ArtifactStore` applies the §3.5 size policy in code, not by convention:
  under `artifacts.commit_under_mb` goes to `artifacts/<run-id>/` and is
  committed; over it goes to Drive keyed by sha256 and only a pointer is
  recorded; over 2 GB is never copied at all. **AT-7 is a test**: a 20 MB file
  produces no bytes in the tree, only a manifest entry.
- `artifacts/<run-id>/manifest.json` records
  `{path, sha256, bytes, store, created_by_step}` per entry, sorted by path, and
  replaces an entry on re-run rather than duplicating it — so re-running a cell
  does not make the manifest describe a file that no longer exists.
- Drive is optional and detected, never required (FR-BOOT-4). With no mount an
  over-cap artifact is **not** silently dropped and not forced into git: it
  becomes an honest pointer whose note says to mount Drive or raise the cap on
  purpose. That is the graceful degradation the PRD asks for.
- Artifact names are untrusted input from cell ids and model arguments, so they
  are sanitised into a single path segment: `../../etc/passwd` becomes
  `passwd.txt`, and a missing extension is inferred from the MIME type.
- `save_artifact` — stores an existing file by `source` or inline `text`.
  It refuses to invent content: an agent that can fabricate a plot it never drew
  is worse than one that reports the cell produced nothing. Empty files are
  rejected as a failed job, not saved as an artifact.
- `display_artifact` — renders image/HTML/text inline (FR-UX-4) through an
  optional injected callback, so the notebook UI stays out of the tool layer.
  Headless (CLI, tests) it references the file instead; a renderer that raises is
  reported and never fails the step. With no path it renders the manifest.
- Notebook integration: `exec_notebook` now stores displayed images and HTML
  automatically, named from the cell id, so re-running replaces the figure
  instead of piling up `output-1.png`, `output-2.png`. Cell streams are not
  artifacts — they are the tail, not the product.
- `save_artifact` joins the commit set, so the artifact and its manifest ride in
  the same commit that produced them, and the `Coagent-Artifacts` trailer names
  only bytes actually in that commit. A Drive pointer is never claimed as a path.
- `Coagent-Artifacts` is now populated end-to-end: `state.artifacts` ← store ←
  the `save_artifact` / `exec_notebook` step, and the manifest is written even
  when every artifact went to Drive or the run stopped early.
- `resume` reads `artifacts/<run>/manifest.json` from git and names the artifacts
  in the briefing, because a fresh VM has the files but no memory of them
  (§3.6). A malformed manifest is ignored rather than blocking a resume.
- The loop now shares one `FileSession` and one `ArtifactStore` across phases, so
  a SURVEY read satisfies ACT's read-before-write guard and artifacts accumulate
  for the whole session.
- `SafetyClass.UI` is now reachable: ACT, VERIFY, and REPORT may render an
  artifact; SURVEY and PLAN may not.

### Notebook UI (`coagent.ui`)
The guaranteed surface (DESIGN §8.1): a chat cell that is good enough that the
product works without Gradio. Split in two on purpose — `render.py` is the
logic and is fully tested; `notebook.py` is plumbing.

- `render.Transcript` folds the loop's event stream into rows. Tool calls become
  collapsed `<details>` blocks — one informative line closed, args and truncated
  output open — so a 40-step run stays readable in a phone-width pane without
  hiding the evidence. HTML in tool output is escaped.
- Headline rendering is per tool, because a collapsed row has to be useful on its
  own: `edit_file src/loader.py +18 −3` counts real changed lines (diff headers
  excluded), a read names the path and size, a shell call reports its status line.
- `render_status_line` is FR-UX-3 in one line: model, context, GPU, session, and
  budget. One line because it sits at the top of every repaint.
- Unknown events are rendered, not dropped, so a new loop event appears in the
  transcript the day it is added rather than the day this file learns of it.
- The row buffer is bounded; old rows leave the *view* but never the session log.
- `ChatCell` draws through `display_id` + `update` rather than
  `clear_output(wait=True)`, which is what §8.1 asks for: the latter flickers and
  scrolls the pane.
- Approvals block on a `threading.Event` a button callback sets — Colab delivers
  widget messages on another thread, so the cell can wait without freezing. The
  wait has a timeout and **denies** on expiry: a missing human is not consent,
  consistent with the non-interactive default (§9.1).
- FR-UX-7 (reconnect-safe): `cell.command("approve"|"deny"|"stop"|"resume")`
  resolves a pending decision from a plain text cell, so a lost widget cannot
  strand a session.
- `ChatCell.render_artifact` draws images and HTML below the transcript, keeping
  the log on screen (FR-UX-4). A renderer that raises is reported, never fatal.
- Everything degrades: no IPython or ipywidgets and the cell still records the
  transcript, and `coAgent()` outside a live kernel stays headless.
- `coAgent()` now wires the cell automatically in a notebook — `on_event`,
  `approver`, and `renderer` — while explicit arguments win and
  `[ui] mode = "headless"` turns it off. That makes the documented two-line
  notebook API (§8.2) actually true.
- `agent.show_log()` — the last N steps with the committed diff (§8.2), read from
  the durable JSONL log so it still works after a resume.
- The loop's `tool` event now carries `path`, so the transcript can name the file
  in a collapsed row without re-parsing tool output.

### Optional Gradio UI (`coagent.ui.gradio_app`)
DESIGN §8.3 calls this a *nicety*, so it is built as one: optional, guarded, and
reusing the same `Transcript` the notebook draws. Nothing in the loop requires it.

- `GradioApp` — a status line, a markdown transcript, an approval banner with
  Approve/Deny, and a prompt box, on `ui.gradio_port` (default 7860).
- Served **loopback only** (`server_name="127.0.0.1"`, `share=False`). In Colab
  the tab reaches it through `serve_kernel_port_as_window`, so there is no reason
  to bind a public interface; ngrok and `share=True` stay out of scope.
- `pip install 'coagent[gradio]'`, or `coagent[all]`. Importing `coagent.ui`
  without Gradio is fine — the widget modules load lazily and `GradioApp()` raises
  `GradioUnavailable` with the exact install command.
- Approvals reuse the notebook contract: a `threading.Event` the loop blocks on,
  a timeout, and **deny on expiry**, because a missing human is not consent (§9.1).
- The pending command is HTML-escaped before it reaches the banner: it is model
  output, and the banner is markup.
- `render_artifact` records the path in the transcript rather than pretending
  Gradio can display an artifact inline; the notebook cell remains the place
  artifacts are actually shown. The two surfaces are honest about the difference.
- Tests substitute a minimal stub when real Gradio is unavailable (it does not
  install on Python 3.15), so the approval queue, hook wiring, HTML escaping, and
  launch kwargs are exercised rather than skipped.

### Tests

475 unit and integration tests, CPU-only — no GPU and no network required.

### Bootstrap (FR-BOOT-1/2, FR-DIS-2)

- `notebooks/bootstrap.ipynb` — the Open-in-Colab landing page. A preflight cell,
  then **exactly three lines** (install, construct, chat), then what the three
  lines hide: the log, the reviewable branch, the resume-after-death cell, and
  the text-command fallback. Tests assert the three-line limit and the
  preflight-before-agent ordering so the landing page cannot silently rot.
- **FR-BOOT-2 is now enforced, not just printed.** `coAgent.__init__` runs the
  T4 preflight *before* constructing the backend and raises `HardwareError` with
  an actionable message, so a CPU runtime fails in a second instead of after a
  four-gigabyte download. `allow_unsupported=True` (and `coagent run
  --allow-unsupported`) is documented in the error itself. An injected backend
  skips the check — that caller has already decided about hardware.
- `coAgent(repo="https://…")` now clones into `coagent-workspace/` instead of
  failing as a bad path, so the documented one-liner is true. Shallow, once:
  an existing `.git` is reused rather than re-cloned. `/content` on Colab, the
  cwd elsewhere.
- **Fixed: a run no longer leaves the tree dirty.** `ArtifactStore.flush()`
  wrote an empty `artifacts/<session>/manifest.json` that nothing committed, and
  the next session then refused to start with `DirtyTreeError` — the resume path
  was broken by a file that recorded nothing. An empty manifest is now not
  written; a real one still is.
Includes end-to-end runs on a real git repository: a scripted model drives
SURVEY → PLAN → ACT → VERIFY → commit with trailers, and separate tests prove
the dirty-tree refusal, plan rejection before any write, honest verify trailers,
and clean budget exhaustion. Notebook execution tests run against a real Jupyter
kernel and skip cleanly when none is installed.

### Release notes — 0.1.0

First release: milestone M1. The loop is proven end-to-end on `FakeBackend` in
`notebooks/bootstrap.ipynb`, and the full v0.1 tool surface from `DESIGN.md` is
built — including `run_tests`/`tools/pyt.py`, which turns pytest output into a
digest instead of a dump. Every module in `DESIGN.md` §11 exists; the remaining
work before a T4 run is breadth of behaviour on a real GPU, not missing code.

Verification performed for this release: 476 tests pass with no skips, and the
built wheel installs into a clean venv where `coagent init`, `config`, `tools`,
and `run --dry-run` all succeed outside the source tree.

Not yet proven: a real 7B NF4 load and a live loop run on a free T4 (acceptance
test AT-1). That requires the Colab runtime and is the first post-release check.
