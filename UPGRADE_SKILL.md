# Engram vLLM upstream upgrade

Use this playbook to update `engram-lab/vllm` without losing fork behavior.

1. Create `jack/<task>` in `~/worktrees/jack/` and add `vllm-project/vllm` as `upstream`.
2. Fetch every fork branch plus `upstream/main`; record both SHAs and run `git fsck --full`.
3. Find the merge base with `git merge-base origin/main upstream/main`; review `git diff --stat <base>..origin/main` and merged fork PRs.
4. Merge pending Engram PRs first so each feature and its tests remain reviewable.
5. Update `AGENT_CODE_CHECKS.md` with one short failure mode per line; describe behavior, not implementation.
6. Merge `upstream/main` with a merge commit; never rebase or squash the shared fork history.
7. Resolve duplicated backports to upstream's version; manually port only Engram-owned cartridges, adapter loading, LoRA, logprob, and model fixes.
8. If upstream moved code, port the behavior into its new abstraction and delete obsolete fork files.
9. Review `git diff upstream/main`, `git diff --check`, conflict markers, renamed paths, imports, and `git fsck --full` before testing.
10. Run Ruff and focused CPU tests for every fork delta; update tests for upstream API changes without weakening assertions.
11. Launch one persistent Modal SSH host with `ENGRAM_USER_ID=jack GPU_COUNT=4 GPU_TYPE=H100` and reuse it for all GPU iterations.
12. Sync with `tools/engram_modal_vllm.sh sync`; use `run` for repeatable remote commands and `shell` for investigation.
13. Run the same pre-commit and focused test commands used by GitHub, then the cartridge regression workflow.
14. Validate Qwen3 and Qwen3.5 with base and LoRA inference across eager/compile and prefix-cache on/off modes.
15. Validate learned prefix cartridges through both chat and completion APIs, including cache miss, cache hit, fallback, reload, and missing-source error paths.
16. Re-run the complete focused suites after every fix; save commands, model IDs, flags, pass counts, and logs.
17. Confirm `git status` is clean, the branch contains current `upstream/main`, and the net fork delta remains feature-focused.
18. Push the branch and report the upstream SHA, retained fork features, removed backports, test matrix, failures fixed, and upgrade highlights.

Prefer tiny public Qwen checkpoints for smoke tests, but use production-equivalent attention, LoRA, compile, and cache settings. A skipped matrix cell is not a pass.
