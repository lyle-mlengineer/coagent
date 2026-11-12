# coAgent — Product Requirements Document

| | |
|---|---|
| **Product** | coAgent (Colab Agent) |
| **Version** | 1.0 (draft) |
| **Date** | 2026-11-12 |
| **Owner** | Lyle Okoth |
| **Status** | Ready for implementation |
| **Design spec** | `DESIGN.md` (architecture and rationale) |
| **Target platform** | Google Colab **free tier — NVIDIA T4 only** |

---

## 1. Summary

coAgent is an AI coding agent that runs entirely inside a Google Colab notebook. The user points it at a GitHub repo they own; it clones the repo, works on a session branch, and uses a locally loaded HuggingFace `transformers` model to plan, edit, verify, and commit code. Every action lands as a reviewable git commit. Artifacts — plots, notebooks, small data files — are rendered inline and saved back to the repo or Drive. If the Colab runtime disappears, the session resumes from its last checkpoint on a fresh VM.

**One sentence:** turn a free Colab T4 into a private, repo-native coding agent with no API keys and no per-token cost.

**Hard constraint for v1:** a single NVIDIA T4 (16 GB) on the free tier, a 7B-class model at 4-bit, and a 12k-token context window. Pro GPUs, TPUs, CPU-only execution, and multi-GPU are explicitly out of scope.

---

## 2. Problem

Cloud coding agents assume an API key, a credit card, and that you are willing to send your code to a third party. That excludes the audience Colab is built for: students, independent researchers, and engineers running experiments on a free GPU. It also puts a meter on exploration — every half-finished attempt costs money.

Colab itself provides the missing compute but none of the workflow: a free T4, a Python REPL, and a disk that is deleted when the session ends. There is no agent that can clone a repo, make a change, run the tests, and leave behind something reviewable.

Data and ML work also break the assumptions of existing agents. The output is often a notebook cell, a plot, or a small artifact — not just source files — and "done" means the cell ran and the figure saved, not only that `pytest` passed.

Two changes make this product possible now: 7B code models are strong enough for narrow, test-verified edits, and git provides durable state and review for free.

**Opportunity:** the first agent that is free, private by construction, and native to both repos and notebooks.

---

## 3. Goals, non-goals, and success

### 3.1 Goals

| ID | Goal | Measured by |
|---|---|---|
| G1 | **Zero-key setup.** No paid API, no account beyond GitHub. | Fresh user reaches a working agent using only a repo URL and a PAT |
| G2 | **Repo-native work.** Every change is a commit on a reviewable branch; `main` is never touched. | Commit trailers, branch protection check |
| G3 | **Survive Colab.** Preemption costs at most one step; resume works on a different VM. | Forced-kill test, §10 |
| G4 | **Verified work.** Nothing counts as done without a passing check. | Definition-of-done enforcement |
| G5 | **Artifacts are first-class.** Plots and notebook outputs render and persist. | Artifact manifest + inline render |
| G6 | **Honest UX.** Plan before edits, approval for risk, never a silent stop. | Approval counts, error-report audit |

### 3.2 Non-goals (v1)

| ID | Non-goal | Rationale |
|---|---|---|
| N1 | Pro GPUs (L4/A100), TPU, CPU-only, multi-GPU | The product is defined by the free tier; supporting more doubles the test matrix |
| N2 | Hosted multi-user service | Colab forbids inbound serving and this is a personal tool |
| N3 | Long training runs | Preemption and VRAM sharing make them unsound; smoke runs only |
| N4 | Model fine-tuning | Out of scope; context engineering and tests dominate the ROI |
| N5 | Multi-repo workspaces | Needs cross-repo retrieval and per-repo locks |
| N6 | Autonomous pushes to `main`, PR merges | Review is the safety model |
| N7 | Image/screenshot input | Local 7B vision models do not fit the budget |
| N8 | macOS/Windows runtime | Colab is Linux only |

### 3.3 Success in one line

A user with no API key can, on a free T4, get a reviewable branch that fixes a real failing test in an unfamiliar 5k-line repo in under 20 minutes of wall clock, and can recover from a killed runtime in under 2 minutes of effort.

---

## 4. Target hardware envelope (T4 only)

### 4.1 Environment

These are observed free-tier limits as of late 2026, not contractual guarantees. The product must degrade predictably if Google changes them.

| Resource | Value | Consequence for the product |
|---|---|---|
| GPU | NVIDIA T4, 16 GB GDDR6 | The only supported accelerator |
| Compute capability | 7.5 (Turing) | fp16 yes; bf16 no; FlashAttention-2 no |
| System RAM | ~12.7 GB | Model + kernel + tests must coexist |
| vCPU | 2 | No parallel decode, no vector DB |
| Local disk | ~70–110 GB, ephemeral | Weights cached locally; nothing durable |
| Session length | up to ~12 h | Tasks must be checkpointed continuously |
| Idle disconnect | ~90 min | Heartbeat, no long silent thinking |
| Inbound ports | none (Colab proxy only) | UI must live in the notebook |
| Outbound network | available, fast (~100 MB/s) | Model download, git push |

### 4.2 Memory plan at 12k context

| Item | Estimate |
|---|---|
| 7B NF4 weights | 4.2–4.6 GB |
| Embeddings / `lm_head` in fp16 | ~1.1 GB |
| KV cache at 12k (28 layers, 4 KV heads) | ~0.7 GB |
| Activations and allocator slack | 1–2 GB |
| **Model resident total** | **~6.5–8 GB** |
| **Headroom for tests / user work** | **≥7 GB** |

### 4.3 Derived hard requirements

| ID | Requirement |
|---|---|
| D1 | Preflight must assert compute capability 7.5, a device name containing `T4`, and ≥14 GB VRAM. On anything else, refuse to start unless `--allow-unsupported` is passed, and never present unsupported runs as validated. |
| D2 | The context window is hard-capped at **12,288 tokens**, counted with the model's own tokenizer. Requests over the cap are rejected before generation, not truncated silently at the server. |
| D3 | Inference uses `float16` and `attn_implementation="sdpa"`. bf16 and FlashAttention-2 must never be assumed. |
| D4 | Verification commands run CPU-only by default (`CUDA_VISIBLE_DEVICES=""`). GPU checks are a separate, explicit step. |
| D5 | Peak VRAM must stay ≤10 GB, leaving headroom for the user's own work. |
| D6 | If the 7B model cannot fit after the OOM ladder, fall back to the 3B model at 8k context; never to a larger model. |

---

## 5. Users

| Persona | Context | Primary need |
|---|---|---|
| **P1 — Solo ML engineer (primary)** | Trains audio/speech models, lives in Colab, keeps code in GitHub | Make repo changes and run checks without leaving Colab or paying per token; keep code private |
| **P2 — Data analyst** | Works in notebooks, pandas, plots | Change code, run a cell, get a figure saved and rendered |
| **P3 — Student / learner** | Free tier only, cannot buy API credits | Use a real coding agent at zero marginal cost |
| **Anti-persona** | Team needing concurrent users, monorepos >50k LOC, someone wanting fully autonomous long-running SWE | Out of scope; would compromise the T4-only design |

---

## 6. User stories

| ID | Story | Acceptance |
|---|---|---|
| US-1 | As a new user, I want to start the agent in one Colab cell without configuring an API key. | Paste repo URL + run cell → agent ready; only secret used is a GitHub PAT from Colab secrets |
| US-2 | As an engineer, I want the agent's work on a branch I can review. | Session branch exists, commits carry trailers, `main` unchanged, diff reviewable |
| US-3 | As a data analyst, I want a plot rendered in the notebook and saved. | Figure displays inline and appears in `artifacts/<run>/` with a manifest hash |
| US-4 | As a user whose session died, I want to resume without redoing work. | New VM + one command → briefing + work continues; ≤1 step lost |
| US-5 | As a cautious user, I want to approve risky actions. | Shell/network/install/delete prompts inline; denial is recorded and respected |
| US-6 | As an engineer, I want a failing test fixed and verified. | Target test goes red→green; full suite green; commit exists |
| US-7 | As a notebook user, I want a cell edited and executed. | Cell updated by id, executed in a subprocess, outputs captured |
| US-8 | As a reviewer, I want to know what the agent did and why. | Plan file, per-step log, commit trailers, and a final report |

---

## 7. Functional requirements

Priorities: **P0** = v1 release blocker, **P1** = v1 if time allows, **P2** = later.

### 7.1 Bootstrap and session lifecycle

**FR-BOOT-1 (P0) — One-cell start.**
The user shall reach a working agent by running a single notebook cell containing at most three lines of user code (install, construct, chat).
*Accept when:* a clean runtime goes from empty notebook to prompt with no other setup.

**FR-BOOT-2 (P0) — Preflight and refusal.**
Before loading weights, the agent shall print GPU name, VRAM, RAM, disk, chosen model, and context budget; and shall refuse to start on hardware failing D1 with an actionable message.
*Accept when:* an A100 or CPU runtime stops with the detected values and the override flag documented.

**FR-BOOT-3 (P0) — Boot budget.**
Cold start (weights not cached) shall reach a prompt in ≤10 minutes; warm start in ≤3 minutes, with visible download/load progress.
*Accept when:* timed on a fresh T4, three runs.

**FR-BOOT-4 (P0) — Model caching.**
Weights shall be cached on local disk, with an optional Drive tarball copied in at boot. The model shall never be memory-mapped from Drive.
*Accept when:* second session in the same VM reuses the cache with no re-download.

**FR-BOOT-5 (P0) — Secrets handling.**
The GitHub token shall be read from Colab secrets at runtime and shall never appear in notebook source, the repo, logs, artifacts, or `.git/config`.
*Accept when:* the redaction corpus test finds zero occurrences in any produced file.

**FR-BOOT-6 (P0) — Optional Drive.**
Mounting Drive shall be an explicit user action. Declining shall degrade gracefully: no Drive bundles or large artifacts, everything else works.
*Accept when:* a run with Drive declined completes a small task and reports the reduced artifact capability.

**FR-BOOT-7 (P1) — Doctor.**
`coagent doctor` shall report versions, GPU, cache state, git auth, and a one-line verdict.

**FR-BOOT-8 (P0) — No background daemons.**
The agent shall run only in the user's interactive kernel and shall not require `nohup`, `screen`, cron, or inbound ports.
*Accept when:* a session runs end-to-end with only the notebook process tree.

### 7.2 Workspace, git, and durability

**FR-GIT-1 (P0) — Clone and refuse dirty.**
The agent shall clone the target repo at depth 1 into `/content/<name>` and shall refuse to start if the working tree is dirty, to avoid clobbering user work.
*Accept when:* a dirty tree produces a stop with instructions, not a reset.

**FR-GIT-2 (P0) — Session branch.**
The agent shall create `coagent/<yyyy-mm-dd>-<short-id>` and shall never commit to or push `main`.

**FR-GIT-3 (P0) — Commit protocol.**
Each verified step shall be one commit with the message and trailers defined in Appendix B (`Coagent-Session`, `Coagent-Step`, `Coagent-Task`, `Coagent-Verify`, `Coagent-Model`, `Coagent-Artifacts`).
*Accept when:* `git log --grep="Coagent-Step"` returns every step, in order.

**FR-GIT-4 (P0) — Durable sync.**
After every commit, the agent shall sync the branch off the VM: push `coagent/*` to a configured remote, or write an incremental `git bundle` to Drive if the push fails or is denied. A failed sync shall warn and continue.
*Accept when:* killing the runtime immediately after a commit loses at most one step after resume.

**FR-GIT-5 (P0) — Resume.**
The agent shall resume from the branch, plan file, session log, and pending bundles on a different VM, printing a briefing first.
*Accept when:* the resume test completes on a fresh runtime with a different IP.

**FR-GIT-6 (P0) — Single-writer lock.**
A session lock with a heartbeat TTL shall prevent concurrent sessions on the same repo. `--force` shall take over after reporting the stale session.

**FR-GIT-7 (P1) — Squash on request.**
The agent shall offer a session-end squash that preserves all trailers.

**FR-GIT-8 (P0) — Remote allowlist.**
Git shall push only to remotes named in config and only to `coagent/*` refs. Any other destination shall be denied before execution.
*Accept when:* a prompt-injected "push to main" attempt is denied and logged.

**FR-GIT-9 (P0) — No destructive history.**
Force-push, `reset --hard`, and branch deletion shall be denied, and deny rules shall evaluate before allow rules.

**FR-GIT-10 (P0) — Artifact policy.**
Files under 5 MB may be committed under `artifacts/<run-id>/`; larger files shall go to Drive and be referenced by SHA-256 in the manifest. The agent shall refuse to commit over-cap binaries.
*Accept when:* a 20 MB output is recorded as a pointer and does not enter git.

### 7.3 Planning and execution

**FR-AGT-1 (P0) — Plan before writing.**
For any task estimated to touch more than one file or more than five steps, the agent shall write `.coagent/plans/<task>.md` before its first write.
*Accept when:* the plan file exists with the first commit that changes code.

**FR-AGT-2 (P0) — Plan approval.**
In interactive mode the plan shall be shown for approve / edit / reject before acting. `--auto` pre-approves. Rejection shall result in no writes.
*Accept when:* a rejected plan produces zero file modifications.

**FR-AGT-3 (P0) — One write per turn.**
A message may contain at most one write or exec action, or up to three read-only actions.
*Accept when:* a malformed multi-write message is rejected with a corrective prompt.

**FR-AGT-4 (P0) — Read before write.**
An edit shall fail if the target path has not been read during the session.
*Accept when:* an edit to an unread file is rejected and the model is told to read it.

**FR-AGT-5 (P0) — Budgets.**
Steps, wall clock, tokens, GPU minutes, and consecutive failures shall be bounded by config; on exhaustion the agent shall checkpoint and report, never continue silently.
*Accept when:* a task capped at 5 steps stops at 5 with a report.

**FR-AGT-6 (P0) — Loop detection.**
An unchanged state fingerprint on three consecutive turns shall force a replan; five shall stop the task with a diagnostic.
*Accept when:* a seeded repeating failure terminates inside the budget.

**FR-AGT-7 (P0) — Verify after write.**
Every write shall be followed by a syntax check, and by the configured lint/test commands before the step may commit. Failures shall be fed back truncated.
*Accept when:* a syntactically invalid edit cannot produce a commit.

**FR-AGT-8 (P0) — Definition of done.**
A task is done only when configured checks pass, every plan item is checked, and no new `TODO` or stub markers were introduced.
*Accept when:* a passing test with an unchecked plan item does not mark the task done.

**FR-AGT-9 (P0) — Ask the user.**
The agent shall have a blocking `ask_user` action that returns free text and resumes the loop.

**FR-AGT-10 (P1) — Codemod.**
For mechanical edits spanning more than three files, the agent should write and run a script instead of many diffs, then verify normally.

**FR-AGT-11 (P1) — Best-of-N.**
After two failures on the same step, the agent may sample up to three candidates and keep the first that passes checks; cost shall be logged.

**FR-AGT-12 (P0) — Final report.**
Every task shall end with a report: what landed, what is blocked, the branch name, artifact list, and the next suggested step.

### 7.4 Model and inference (T4)

**FR-MDL-1 (P0) — Model set.**
The default model shall be `Qwen/Qwen2.5-Coder-7B-Instruct` at NF4; the only fallback is `Qwen2.5-Coder-3B-Instruct` at 8k. No other model is validated in v1.

**FR-MDL-2 (P0) — Context cap.**
Every generation request shall be assembled against the 12k budget table and rejected if over cap.

**FR-MDL-3 (P0) — T4-compatible inference.**
fp16 + SDPA on T4; no bf16, no FlashAttention-2, no `torch.compile`.

**FR-MDL-4 (P0) — Sampling.**
Edits at temperature 0.2; planning at 0.7; a per-session seed shall be logged.

**FR-MDL-5 (P0) — Streaming.**
Tokens shall stream to the UI with a stop control; a cancelled generation shall not corrupt state.

**FR-MDL-6 (P0) — OOM ladder.**
CUDA OOM shall trigger, in order: cache clear + 25% less context, halved output, 3B fallback, stop with a report. The session shall not be lost.

**FR-MDL-7 (P0) — No inference egress.**
Model execution shall make no network calls. The only outbound traffic is the model download and git.

**FR-MDL-8 (P1) — Constrained decoding.**
JSON-envelope constrained decoding may be enabled by config, off by default, and must not remove the tolerant-parse path.

**FR-MDL-9 (P0) — Download resilience.**
Model download shall resume after interruption and verify via the library's checksums.

**FR-MDL-10 (P0) — Token accounting.**
Every turn shall log input/output tokens and wall time; the UI shall show cumulative session usage.

### 7.5 Context

**FR-CTX-1 (P0) — Repo map.**
A signature-only repo map shall be generated at session start, capped at 2k tokens, updated incrementally after edits.

**FR-CTX-2 (P0) — Retrieval.**
BM25 over function-sized chunks plus symbol adjacency, with no vector database in v1.

**FR-CTX-3 (P0) — Deterministic truncation.**
Observations shall be head/tail truncated with line numbers, never silently dropped mid-file.

**FR-CTX-4 (P0) — Notebook outputs excluded.**
Cell outputs shall go to artifacts; only the last traceback or a 40-line tail may enter context.

**FR-CTX-5 (P0) — Plan survives eviction.**
After any context eviction, the plan file shall be re-read before the next action.

**FR-CTX-6 (P0) — Tokenizer truth.**
All budget arithmetic shall use the loaded model's tokenizer, not character heuristics.

### 7.6 Tools, safety, and audit

**FR-TOL-1 (P0) — Registry.**
Only registered tools with schemas may execute; unknown tools and unknown arguments shall be rejected with a corrective message.

**FR-TOL-2 (P0) — Path jail.**
Every path shall resolve inside the repo root, `/tmp/coagent`, or an explicit `allow_read` entry. Reads are jailed too.

**FR-TOL-3 (P0) — Process isolation.**
Exec actions shall run in subprocesses with a fresh session, timeout, process-group kill, scrubbed environment, and a 64 KB output cap.

**FR-TOL-4 (P0) — Approvals.**
The approval matrix shall be enforced in code: reads inside the jail allowed; writes on the session branch allowed; shell, network, install, delete, and outside-repo writes require approval in interactive mode and are denied by default otherwise.

**FR-TOL-5 (P0) — Denylist.**
Destructive patterns shall be denied before execution; the documentation shall state that this is a seatbelt, not a sandbox.

**FR-TOL-6 (P0) — Notebook tools.**
Notebook edits shall address cells by id; execution shall run via `nbconvert` in a subprocess with a 300 s timeout.

**FR-TOL-7 (P0) — Artifact tools.**
`display_artifact` shall render image/HTML in the cell; `save_artifact` shall apply the size policy and manifest.

**FR-TOL-8 (P0) — Audit log.**
Every tool call shall be appended to the session log with arguments hash, outcome, tokens, and duration; secrets shall be redacted at write time.

**FR-TOL-9 (P1) — Codemod sandbox.**
Codemod scripts shall run from `.coagent/tmp/` with the repo on `PYTHONPATH`, under the same exec safety class.

### 7.7 Verification

**FR-VER-1 (P0) — CPU tests by default.**
`verify.commands` shall run with the GPU hidden.

**FR-VER-2 (P0) — GPU checks once.**
`verify.gpu_commands` shall run once per task, after unloading the model, checking free VRAM against `vram_reserve_mb`, and reloading afterwards.

**FR-VER-3 (P0) — Failure extraction.**
Test feedback shall be reduced to failing test names plus the tail of each traceback, capped at ~1500 tokens.

**FR-VER-4 (P0) — Syntax gate.**
No edit may count as a step until it parses in its language.

**FR-VER-5 (P0) — Tests required when absent.**
If `require_tests` is set and the repo has no relevant test, the task plan must include writing one before it can be done.

### 7.8 User experience

**FR-UX-1 (P0) — Chat cell.**
Streaming responses, collapsible tool steps showing args and truncated output, and a visible diff per edit.

**FR-UX-2 (P0) — Plan surface.**
The plan renders with per-item status and approve / edit / reject controls.

**FR-UX-3 (P0) — Status line.**
Model, context budget, GPU, VRAM used, session id, and cumulative tokens shall be visible at all times.

**FR-UX-4 (P0) — Inline artifacts.**
Rendered figures and saved-file links appear in the transcript.

**FR-UX-5 (P0) — Actionable errors.**
Every failure shall name the cause, the affected file or command, and one suggested next action. No silent stops.

**FR-UX-6 (P1) — Rich UI.**
Optional Gradio app exposed via Colab's port proxy; the notebook UI must remain fully functional without it.

**FR-UX-7 (P0) — Reconnect-safe.**
If widget state is lost, the agent shall accept text commands (`approve`, `deny`, `stop`, `resume`) so the session is not stuck.

**FR-UX-8 (P0) — Interrupt.**
A stop control shall cancel generation or a running command without losing the checkpoint.

### 7.9 Reliability and recovery

**FR-REL-1 (P0) — Heartbeat.**
While the tree is dirty, the agent shall commit and sync at least every 3 minutes and on every verified step.

**FR-REL-2 (P0) — Signal handling.**
`SIGTERM`, `SIGINT`, and kernel shutdown shall attempt a checkpoint before exit.

**FR-REL-3 (P0) — Crash budget.**
A hard kill shall lose at most one step.

**FR-REL-4 (P0) — Restart detection.**
After a runtime restart, one command shall reinstall, reload, and resume from the last synced step.

**FR-REL-5 (P0) — No outside writes.**
The agent shall never modify files outside the repo, `/tmp/coagent`, and the Drive artifact directory.

### 7.10 Privacy

**FR-PRV-1 (P0) — No code egress.**
Source code and prompts shall never leave the VM except through git pushes to the configured remote.

**FR-PRV-2 (P0) — Redaction.**
GitHub, HuggingFace, cloud, and JWT-shaped secrets shall be redacted in every log, display, report, and committed file.

**FR-PRV-3 (P0) — Clean logs.**
Committed session logs shall contain no credentials, verified by the redaction corpus test.

**FR-PRV-4 (P1) — Offline mode.**
A bundle-only mode shall work with git push disabled entirely.

### 7.11 Distribution and documentation

**FR-DIS-1 (P0) — Install.**
`pip install "git+https://github.com/<owner>/coagent"` with pinned dependencies.

**FR-DIS-2 (P0) — Bootstrap notebook.**
A single notebook with an "Open in Colab" badge is the landing page.

**FR-DIS-3 (P0) — Demo.**
README with a <5-minute recorded demo: clone → task → green tests → branch, plus a resume demo.

**FR-DIS-4 (P1) — Diagnostics.**
`coagent doctor` and `coagent report` (session summary from the log).

**FR-DIS-5 (P0) — Version compatibility.**
The package shall refuse to run against an incompatible `.coagent/config.toml` schema and print the migration step.

---

## 8. Non-functional requirements

| ID | Area | Requirement |
|---|---|---|
| NFR-1 | Latency | Turn p50 ≤90 s, p95 ≤180 s on T4 excluding human approval wait |
| NFR-2 | Boot | Cold ≤10 min, warm ≤3 min |
| NFR-3 | Memory | Model ≤8 GB VRAM; peak ≤10 GB; system RSS ≤10 GB |
| NFR-4 | Context | No request exceeds 12,288 tokens |
| NFR-5 | Recovery | Resume to first token ≤10 min after a kill, including model load |
| NFR-6 | Security | Zero secrets in produced files; non-interactive mode defaults to deny |
| NFR-7 | Compatibility | Colab Linux, Python 3.11, CUDA 12.x, pinned `transformers` |
| NFR-8 | Observability | JSONL metrics per step; `coagent report` builds a session summary |
| NFR-9 | Accessibility | Keyboard-operable controls, AA contrast, status never conveyed by color alone |
| NFR-10 | Cost | Zero marginal cost beyond the free tier; no paid API calls anywhere |

---

## 9. UX flows

### 9.1 First run (target: 5 min warm, ≤10 min cold)

1. Open the bootstrap notebook, run the cell.
2. Paste repo URL; the PAT is read from Colab secrets (no paste into the cell).
3. Preflight prints `T4 · 16 GB · 7B NF4 · 12k ctx` and refuses anything else.
4. Weights download/load with progress.
5. Repo clones, `main` checked out read-only, session branch created.
6. The prompt appears with an empty plan and a status line.

### 9.2 A task

1. User types the task.
2. Survey: repo map + relevant files, no writes.
3. Plan shown; user approves.
4. Act → verify loop, one change per turn, streaming.
5. Each verified step commits and syncs.
6. Report: branch, diff stat, artifacts, next step.

### 9.3 An approval

Risky action attempted → inline card (`command`, `reason`, `[Allow once] [Always] [Deny]`) → decision is logged, the command runs or the model is told it was denied.

### 9.4 Recovery

New VM → bootstrap cell → `agent.resume()` → briefing (`last commit, plan state, open failure`) → continue. If Drive bundles are ahead of the remote, replay them first.

### 9.5 Export

`agent.push()` pushes the branch and prints the compare URL. Artifacts are downloadable from the transcript or listed in the manifest.

---

## 10. Metrics, acceptance tests, and release gates

### 10.1 Product metrics

| Metric | Target | Method |
|---|---|---|
| Task success | ≥70% | 20-task eval, 3 runs, majority, tests as truth |
| Median turns-to-done | ≤12 | Session logs |
| p50 task wall clock | ≤20 min | Logs |
| Crash loss | ≤1 step, 10/10 | Forced-kill test |
| Secret leakage | 0 | Redaction corpus over all outputs |
| Approvals per successful task | ≤3 | Logs |
| First task, warm start | ≤15 min | Timed demo |
| Peak VRAM | ≤10 GB | `nvidia-smi` sampler during eval |
| Regression rate | <10% | Eval diff between releases |

### 10.2 Acceptance tests

| ID | Test |
|---|---|
| AT-1 | Fresh T4, cold start, fixed bug in toy repo, green tests, branch exists |
| AT-2 | `kill -9` between two steps → resume on a new VM, ≤1 step lost |
| AT-3 | Dirty worktree at start → refuse, user files intact |
| AT-4 | Push denied (no network) → bundle written, resume replays it |
| AT-5 | GPU test configured → model unloads, test runs, model reloads |
| AT-6 | Prompt injection in README requesting `git push origin main` → denied and logged |
| AT-7 | 20 MB artifact → pointer + manifest, not in git |
| AT-8 | Recipe containing a fake PAT → redacted everywhere |
| AT-9 | Rejected plan → zero writes |
| AT-10 | Notebook cell that loops forever → killed at 300 s, session survives |
| AT-11 | Context over cap → rejected before generation, no crash |
| AT-12 | 10 consecutive forced kills → all resumes succeed |

### 10.3 Release gates

| Release | Required |
|---|---|
| v0.1 | AT-1, AT-8; single-file task end-to-end on T4 |
| v0.2 | AT-2, AT-3, AT-4; durability and resume |
| v0.3 | AT-5, AT-6, AT-9; plan gate, approvals, artifacts; 5-task eval ≥60% |
| v1.0 | All ATs; 20-task eval ≥70%; demo + docs; zero open P0 bugs |

---

## 11. Milestones

| Release | Week | Scope | Exit criterion |
|---|---|---|---|
| v0.1 | 1 | Loop, six tools, transformers backend, notebook chat | AT-1 passes |
| v0.2 | 2 | Repo map, SEARCH/REPLACE reliability, verify loop, durable sync, resume | AT-2/3/4 pass |
| v0.3 | 3 | Plan approval, artifacts, approvals UI, Drive cache, OOM ladder | AT-5/6/9 pass; 5-task eval ≥60% |
| v1.0 | 4 | Constrained decoding (flagged), evals, docs, demo, T4 matrix | All ATs; 20-task eval ≥70% |

Definition of shipped: a stranger with a free Colab account and a repo URL can install, run a task, review the branch, and recover from a disconnect using only the README.

---

## 12. Dependencies and assumptions

| Type | Item | Note |
|---|---|---|
| Model | Qwen2.5-Coder-7B-Instruct (Apache-2.0) | 3B fallback |
| Runtime | `transformers`, `bitsandbytes`, `accelerate`, `torch` | Pinned; CUDA 12.x |
| Parsing | `tree-sitter` + per-language grammars | Regex fallback if a wheel is missing |
| Notebooks | `nbformat`, `nbconvert` | Execution in subprocess |
| UI | `ipywidgets`; `gradio` optional | Gradio behind Colab proxy only |
| External | GitHub fine-grained PAT | Scoped to one repo |
| External | HuggingFace Hub | Download only |
| Assumption | A free T4 is available when the user starts | Product refuses clearly when not |
| Assumption | User owns the target repo and can push a branch | |
| Assumption | Outbound network is available | Offline mode is P1 |
| Assumption | Colab limits may change without notice | Requirements degrade predictably |
| Constraint | Colab ToS: interactive use, no persistent serving | No daemons, no tunnels |
| Constraint | No paid services anywhere in the default path | Zero marginal cost |

---

## 13. Risks

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | 7B quality is below the bar | Product feels unreliable | Narrow tasks, tests-as-truth, best-of-N; if it still fails, cut scope to single-file tasks |
| R2 | No T4 available at peak times | User cannot start | Clear message, retry guidance; never silently degrade to CPU |
| R3 | Colab policy or limits change | Breakage | No daemons; configurable budgets; heartbeat not tied to exact 12 h |
| R4 | Preemption mid-task | Lost work | Durable sync every checkpoint; ≤1 step loss |
| R5 | Dependency breakage (outlines, tree-sitter, bnb) | Install fails | Pin versions; tolerant-parse default; regex repo-map fallback |
| R6 | Prompt injection via repo or output | Unauthorized action | Approvals, remote allowlist, branch isolation; never claim prompt text is a defense |
| R7 | WIP history leaked on a public repo | Privacy | Private `durable` mirror; push-on-verify option |
| R8 | Model download stalls | Bad first run | Resume support, Drive tarball |
| R9 | GPU tests OOM against the model | Failed verification | Model-yields-GPU step, VRAM reserve, CPU default |
| R10 | Context overflow mid-task | Garbled edits | Enforced cap, eviction order, plan re-read |

---

## 14. Decisions and open questions

### 14.1 Decision log

| ID | Date | Decision | Rationale |
|---|---|---|---|
| D-1 | 2026-11-12 | **Free-tier T4 is the only supported GPU.** Pro/TPU/CPU out of scope. | Product definition; keeps the test matrix and every budget in this PRD valid |
| D-2 | 2026-11-12 | Notebook-first surface | The product lives where the free GPU is |
| D-3 | 2026-11-12 | One repo per session | Avoids cross-repo retrieval and multi-lock complexity |
| D-4 | 2026-11-12 | No long training; scripts and ≤5 min smoke runs only | Preemption and shared VRAM make training unsound |
| D-5 | 2026-11-12 | Reads are repo-only plus explicit `allow_read` | Protects the user's Drive after mount |
| D-6 | 2026-11-12 | Durable sync on every checkpoint; push default, bundle fallback | Makes the one-step crash budget true |
| D-7 | 2026-11-12 | Plan approval required for multi-file tasks | Prevents minutes of GPU on a misunderstood task |

`DESIGN.md` §4.1 now reflects D-1; the design's other budgets are unchanged.

### 14.2 Open questions

| ID | Question | Owner | Needed by |
|---|---|---|---|
| Q-1 | What is the numeric baseline for turns-to-done (human time, cloud-agent reference)? | Product | v0.3 |
| Q-2 | Ship on PyPI, or git URL only for v1? | Product | v1.0 |
| Q-3 | Final WIP-push policy for public repos: private mirror required, or opt-in? | Product | v0.2 |
| Q-4 | Does the optional Gradio UI make v1, or v1.1? | Product | v0.3 |
| Q-5 | Fallback model if Qwen2.5-Coder becomes unavailable or changes license? | Eng | v1.0 |
| Q-6 | Is any telemetry acceptable? (Current assumption: none.) | Product | v0.1 |

---

## 15. Appendix

### A. Configuration reference

See `DESIGN.md` §11 for the full `.coagent/config.toml`. The keys this PRD depends on: `model.max_context` (D2), `verify.commands`, `verify.gpu_commands`, `verify.vram_reserve_mb`, `git.remotes`, `durability.mode`, `tools.allow_read`.

### B. Commit trailer spec

```
Coagent-Session: <session-id>
Coagent-Step: <n>
Coagent-Task: <task text, one line>
Coagent-Verify: <command summary and result>
Coagent-Model: <model>@<quant>
Coagent-Artifacts: <comma-separated paths>
```

### C. Glossary

| Term | Meaning |
|---|---|
| Turn | One model message plus its tool execution |
| Step | A turn that produced and verified a change, committed |
| Checkpoint | Commit + durable sync |
| Plan file | `.coagent/plans/<task>.md`, the memory that survives eviction and preemption |
| Repo map | Signature-only symbol index used instead of exploration |
| Jail | Resolved-path containment for reads and writes |
| Done criteria | Passing checks + completed plan + no new stubs |
| Durable sync | Pushing or bundling history off the VM so a kill costs ≤1 step |

### D. Traceability

| Requirement group | Design reference | Acceptance tests | Release |
|---|---|---|---|
| Bootstrap (FR-BOOT) | `DESIGN.md` §11 | AT-1, AT-8 | v0.1 |
| Workspace/durability (FR-GIT) | `DESIGN.md` §3 | AT-2, AT-3, AT-4, AT-7 | v0.2 |
| Planning/execution (FR-AGT) | `DESIGN.md` §7 | AT-1, AT-9, AT-12 | v0.1–v1.0 |
| Model (FR-MDL) | `DESIGN.md` §4 | AT-11 | v0.1 |
| Context (FR-CTX) | `DESIGN.md` §5 | AT-11 | v0.1 |
| Tools/safety (FR-TOL) | `DESIGN.md` §6, §9 | AT-5, AT-6, AT-7, AT-10 | v0.3 |
| Verification (FR-VER) | `DESIGN.md` §7.5 | AT-5 | v0.2 |
| UX (FR-UX) | `DESIGN.md` §8 | AT-1, AT-9 | v0.3 |
| Reliability (FR-REL) | `DESIGN.md` §3.6, §10 | AT-2, AT-4, AT-12 | v0.2 |
| Privacy (FR-PRV) | `DESIGN.md` §9.2 | AT-6, AT-8 | v1.0 |
| Distribution (FR-DIS) | `DESIGN.md` §11 | AT-1 | v1.0 |
