---
name: trace-engram-stack
description: Trace architecture, source-control provenance, runtime data flow, and change ownership across engram-lab/engrams, engram-lab/torchtitan, and engram-lab/vllm. Use when explaining how the repositories connect, deciding where a cross-repo change belongs, checking fork or upstream divergence, syncing or porting TorchTitan or vLLM work, debugging adapter and checkpoint compatibility, or preparing integration tests across these repositories.
---

# Trace the Engram stack

Build the explanation from the checked-out code and Git refs. A branch name says which ref to follow, but it does not prove that two trees match.

## Establish the current state

1. Read the instructions in each repository that the task touches.
2. Run `scripts/inspect-stack.sh`. Pass the three repository roots if they are not sibling directories.
3. Check `git remote -v` before fetching. A local `upstream/main` ref can exist without an `upstream` remote. If the answer depends on the latest remote state, refresh from the verified upstream URL. Otherwise, label remote-tracking refs as locally cached.
4. Read [references/repo-map.md](references/repo-map.md). Verify any path involved in the task because branches move.

Use this command from any of the three repository roots:

```bash
.agents/skills/trace-engram-stack/scripts/inspect-stack.sh
```

Use explicit paths when the checkouts have other names:

```bash
.agents/skills/trace-engram-stack/scripts/inspect-stack.sh \
  /path/to/engrams /path/to/torchtitan /path/to/vllm
```

## Pick the owner before editing

Use these ownership defaults, then confirm them by tracing the symbols involved:

| Concern | Start in | Integration branch |
| --- | --- | --- |
| Orchestration, clients, deployment, training-server adapters, and cross-repo contract tests | `engram-lab/engrams` | `origin/dev` |
| Apr28 model, trainer, checkpoint, parallelism, and numeric behavior | `engram-lab/torchtitan` | `origin/engram/torchtitan_apr_28_26` |
| Inference scheduling, LoRA loading, prefix cartridges, model execution, and OpenAI-compatible serving | `engram-lab/vllm` | `origin/main` |
| The embedded trainer under `libs/torchtitan` | `engram-lab/engrams` | `origin/dev` |

Do not confuse the embedded `libs/torchtitan` tree with the external Apr28 checkout. A fix may need paired changes, but the two trees have different histories and tests.

## Trace a cross-repository behavior

1. Find the user-facing entry point in Engrams.
2. Follow the exact config, environment variable, request field, or artifact path across the repository boundary.
3. Read the receiving implementation in TorchTitan or vLLM.
4. Record the wire or file contract. Include field names, tensor names and shapes, checkpoint layout, or CLI flags when they matter.
5. Locate producer tests and consumer tests. A boundary change is incomplete if only one side is covered.
6. Record the exact commits used by a run. Prefer immutable SHAs over moving branch tips for reproductions.

For adapter flows, start with `SavedAdaptersConfig.to_vllm_dict()` in Engrams and `EngramAdapterServingMixin` plus `vllm.v1.cartridge_loader` in vLLM. For Apr28 training, start with `Apr28TorchTitanRef`, `TorchTitanApr282026Backend`, and the isolated worker bootstrap in Engrams, then follow imports into the selected TorchTitan checkout.

## Explain the connection

Include only sections that help answer the question:

- **State.** Name the checked refs and SHAs. Distinguish the working branch, `origin`, and `upstream`.
- **Ownership.** Say which repository owns each behavior and why.
- **Flow.** Walk from the Engrams caller to the TorchTitan or vLLM implementation and back to the observable result.
- **Source history.** State whether code is embedded, cloned at runtime, pinned by SHA, or inherited from upstream.
- **Gotchas.** Call out stale refs, dirty local-source builds, duplicated implementations, and compatibility assumptions.
- **Evidence.** Cite concrete files, symbols, tests, and Git commands.

Do not publish static ahead or behind counts without the compared SHAs. Do not say a fork is synced because its branch is named `main`.

## Make a cross-repository change

1. Base Engrams work on `origin/dev`.
2. Base Apr28 TorchTitan work on `origin/engram/torchtitan_apr_28_26`.
3. Base Engram vLLM work on `origin/main`, then compare it with `upstream/main` before porting an upstream change.
4. Change the repository that owns the behavior. Add the smallest consumer change needed to use the new contract.
5. Run the owner's focused tests, then the consumer's cross-repository tests.
6. Pin both repository SHAs in remote or GPU validation and report them together. Reciprocal branch-tip CI does not create an atomic pair for two unmerged changes.

For a vLLM upstream merge, also read `UPGRADE_SKILL.md` and `AGENT_CODE_CHECKS.md` in the vLLM repository. Follow the current user's branch and identity rules instead of the playbook's old `jack/` branch example. For TorchTitan integration, read `engram/docs/ci_design.md`, `ci/README.md`, and the active Engram workflows. For Engrams changes, read `AGENTS.md` and `AGENTS_CODE_CHECKS.md`.
