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
- `coagent` CLI with `init`, `doctor`, and `config` verbs.
- `coagent doctor` reports the GPU envelope and whether it is a free-tier T4.
- Dev extras (`[llm]`, `[ui]`, `[context]`, `[colab]`, `[dev]`) so the package
  installs and tests on CPU without torch.

## [0.1.0] - TBD

First release target: milestone M1. Fix a failing test in a toy repo end-to-end
on a free T4 (acceptance test AT-1).
