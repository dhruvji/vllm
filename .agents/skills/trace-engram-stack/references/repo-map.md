# Repository map

Use this page as a starting model. Verify paths and refs in the current checkouts before making a claim.

## Source-control relationships

```text
pytorch/torchtitan upstream/main
    |-- subtree imports and Engrams-local commits --> engram-lab/engrams:origin/dev/libs/torchtitan
    `-- fork history --> engram-lab/torchtitan:origin/engram/torchtitan_apr_28_26
                            `-- selected with TORCHTITAN_DIR --> Engrams Apr28 backend

vllm-project/vllm upstream/main
    `-- periodic merges plus Engram-owned commits --> engram-lab/vllm:origin/main
                                                      `-- exact image commit --> Engrams inference
```

Engrams uses two TorchTitan versions:

- `libs/torchtitan` is an ordinary tracked directory on `origin/dev`. Its history contains `git-subtree-dir: libs/torchtitan` imports plus later Engrams commits. The checked-in `.gitmodules` entry does not make the path a live submodule. Verify with `git ls-tree <ref> libs/torchtitan`: mode `040000` is a directory tree, while a submodule would use mode `160000`.
- The Apr28 backend selects a separate checkout from `engram-lab/torchtitan`, normally branch `engram/torchtitan_apr_28_26`. Engrams tests and remote recipes can pin a full commit SHA. Worker startup puts `TORCHTITAN_DIR` first on `PYTHONPATH` and rejects an import from another checkout.

Do not copy a change between these TorchTitan versions without checking whether both runtimes need it.
The Apr28 name identifies the upstream fork point. The branch continues to receive Engram commits and is not a frozen April snapshot.

## Engrams owns the integration

Use `engram-lab/engrams:origin/dev` for integration work. The main boundaries are:

- `libs/train/train/server/torchtitan_apr_28_26/ref.py` defines the default external TorchTitan repository, branch, commit override, and checkout path.
- `libs/train/train/server/torchtitan_apr_28_26/backend.py` launches the isolated worker and passes the selected checkout through `TORCHTITAN_DIR` and `PYTHONPATH`.
- `libs/train/train/server/torchtitan_apr_28_26/worker_bootstrap.py` activates the checkout before importing worker dependencies.
- `libs/train/train/server/torchtitan_apr_28_26/worker_checkpoints.py` converts selected TorchTitan state into inference-ready PEFT files.
- `libs/train/train/server/engine.py` selects either the embedded `torchtitan` backend or the external `torchtitan_apr_28_26` backend.
- `.github/workflows/ci-train-torchtitan-apr28.yml` checks the branch tip of the external TorchTitan fork against Engrams-owned backend tests.
- `libs/core/core/servers/vllm/config.py` defaults Engram inference images to `engram-lab/vllm` branch `main`.
- `libs/core/core/servers/vllm/image.py` resolves the fork commit and calls the canonical image builder.
- `libs/core/core/infra/modal/base_images.py` builds from `engram-lab/vllm`. A set `VLLM_REPO_ROOT` uses a local source overlay and includes a dirty-diff fingerprint; otherwise the builder clones the configured branch and checks out the resolved commit.
- `libs/core/core/clients/base.py` defines the request-body adapter contract. `libs/core/core/clients/vllm/client.py` sends it as `extra_body["adapters"]` while keeping `model` equal to the base model.
- `libs/core/core/clients/vllm/client.py` and `helpers.py` also consume fork-specific token-ID and prompt-logprob response fields used by training and evaluation.
- `libs/core/core/infra/modal/volumes.py` resolves published adapter paths for S3, Modal volume, or local loading.

`StockVLLMImageConfig` is an explicit upstream-package comparison path. It is not the default, and fork-only adapter, token-ID, and logprob fields must not be assumed to work there.

`libs/core/core/vllm_build_config.py` also pins the precompiled native wheel commit. That binary-compatibility pin is separate from the dynamically resolved source commit on `engram-lab/vllm:main`; report both when a native image matters.

## TorchTitan owns training semantics

Use `engram-lab/torchtitan:origin/engram/torchtitan_apr_28_26` for the external Apr28 runtime. Compare that ref with `upstream/main` before assuming an upstream API exists in the fork.

Do not use the fork's `origin/main` as a substitute for PyTorch upstream. Verify the URL behind each remote or fetch the upstream URL into a clearly named ref before comparing.

Start with these paths:

- `torchtitan/trainer.py` for the training lifecycle.
- `torchtitan/components/peft/` for LoRA ownership, sharding, and gradient behavior.
- `torchtitan/components/checkpoint.py` and model state-dict adapters for artifact layouts.
- `torchtitan/models/qwen3_5/` and `torchtitan/models/deepseek_v3/` for the model families used by the Engrams bridge.
- `engram/docs/ci_design.md`, `ci/README.md`, and `.github/workflows/engram-*.yml` for native, GPU, and cross-repository checks.

The Engrams Apr28 worker records the selected TorchTitan commit in its ready and heartbeat payloads. Use that value when debugging a run.

TorchTitan also runs a reciprocal integration workflow against `engram-lab/engrams@dev`. The Engrams workflow tests its current revision against the Apr28 branch tip. These two workflows catch compatibility drift, but neither validates two unmerged commits as one atomic pair.

## vLLM owns inference semantics

Use `engram-lab/vllm:origin/main` for Engram inference. This branch periodically merges `vllm-project/vllm:upstream/main`, but it also retains Engram-owned behavior. Always compute the merge base and both sides of the divergence before an upgrade.

The most important fork contracts are:

- `vllm/entrypoints/generate/adapter_serving.py` accepts request-body LoRA specs, resolves ranked sources, stages remote artifacts, and calls `engine_client.add_lora()`.
- `vllm/utils/adapter_manager.py` loads and caches adapter artifacts from the supported paths.
- `vllm/v1/cartridge_loader.py` accepts learned prefix cartridges and validates TorchTitan's flat KV tensor layout: `layers.{N}.attention.prefix.keys|values` with shape `(1, seq_len, num_kv_heads, head_dim)`.
- OpenAI chat and completion serving paths carry `adapters` into the engine.
- `AGENT_CODE_CHECKS.md` lists fork behavior that an upstream merge must preserve.
- `UPGRADE_SKILL.md` defines the fork's upstream-merge procedure and test matrix.

The vLLM fork may have a locally fetched `upstream/main` ref without an `upstream` remote. Verify `git remote -v` before refreshing it. Compare `origin/main` with the verified `vllm-project/vllm` ref, not with unrelated branches visible through `git log --all`.

Adapter loaders dispatch from the URI or path in `id`, not from the descriptive `source` field. S3, Modal volume, and local paths are the proven request-body paths in the current fork. Verify other source labels end to end before claiming support.

`adapters` currently crosses the upstream OpenAI protocol as an allowed extra field, and fork code reads it from the request object. Treat upstream changes to protocol-model extra-field handling as a compatibility risk even when the adapter loader itself has not changed.

## Runtime flow

1. An Engrams recipe selects a training backend and an exact or moving TorchTitan ref.
2. Engrams prepares the checkout, starts the isolated worker, and sends training operations across its worker protocol.
3. TorchTitan computes updates and writes checkpoints or adapters in the agreed layout.
4. Engrams resolves and publishes the artifact, then sends an OpenAI-compatible request with `extra_body["adapters"]`.
5. The Engram vLLM fork resolves the LoRA or prefix source, loads it into the inference engine, and serves the request against the base model.

The API and artifact boundaries belong to both sides. When a field name, module alias, tensor shape, or checkpoint key changes, test the producer and the consumer at pinned commits.

## Commands that answer common questions

Run these only after confirming the remote names:

```bash
# Is a fork ahead of or behind its upstream ref?
git merge-base <fork-ref> <upstream-ref>
git rev-list --left-right --count <fork-ref>...<upstream-ref>

# Which commits remain fork-only after patch equivalence?
git log --cherry-pick --left-only --oneline <fork-ref>...<upstream-ref>

# Is Engrams' TorchTitan path a directory or a submodule at this ref?
git ls-tree <engrams-ref> libs/torchtitan

# Which upstream TorchTitan split was most recently imported into Engrams?
git log <engrams-ref> -1 --format=full --grep='git-subtree-dir: libs/torchtitan'
```

Ahead and behind counts describe the two named SHAs at one moment. Include both SHAs when reporting the counts.
