# Selection Method Adaptation Audit

This note records what is actually reusable from each upstream repository for
ECG+PPG data selection. The project must not call an ECG+PPG migration result an
original-paper reproduction unless the original dataset, split, model, training
script, and metric are also used.

## Result Labels

| Label | Meaning |
| --- | --- |
| `ORIGINAL_REPRODUCTION` | Run the upstream repository on the original paper dataset, protocol, model, and metric. |
| `OFFICIAL_WORKFLOW_ADAPTED` | Keep the upstream training/selection workflow, but replace dataset/model adapters for ECG+PPG. |
| `OFFICIAL_CORE_ADAPTED` | Use the upstream score or selector implementation, but use our ECG+PPG training loop or embeddings. |
| `CONCEPTUAL_EXTENSION` | Only the high-level idea transfers; this is not a reproduction or official-code benchmark. |

## Audit Table

| Method | Upstream | Code contract found in repo | ECG+PPG adaptation decision | Label for our benchmark |
| --- | --- | --- | --- | --- |
| ELFS | `third_party/elfs` @ `089444b` | Image datasets, DINO/ResNet-style embedding generation, pseudo-label proxy training dynamics, then `core/data/Coreset.py` score-based selection. | Do not fake ECG/PPG as images. Build an ECG+PPG proxy encoder, cluster pseudo-labels from ECG/PPG embeddings, collect ELFS-compatible training-dynamics scores, and call the official coreset score selector. | `OFFICIAL_CORE_ADAPTED` |
| PreSelect | `third_party/PreSelect` @ `258c632` | Text/LLM pipeline: BPC from causal LMs, fastText predictive scorer, DataTrove/Megatron-style filtering and training. | Not a native ECG+PPG coreset method. Using it on waveform windows would require designing a new predictive-loss proxy and scorer. Keep out of the ECG+PPG benchmark unless explicitly marked as a conceptual extension. | `CONCEPTUAL_EXTENSION` |
| Moderate Coreset | `third_party/2023_ICLR_Moderate-DS` @ `7f64d31` | `selection.py` selects samples by distance to per-class feature medians; official scripts extract image features from CIFAR/Tiny ImageNet checkpoints. | Feed ECG+PPG embeddings and task labels or fixed pseudo-label bins into the official median-distance selector. Downstream ECG+PPG training stays in our protocol. | `OFFICIAL_CORE_ADAPTED` |
| GLISTER | `third_party/GLISTER` @ `7b70e18`; also available through `third_party/cords` | Original code supports NumPy custom datasets and active-learning loops; CORDS provides PyTorch DataLoader/model/loss strategy classes. | Prefer CORDS for ECG+PPG because it accepts DataLoader, model, loss, validation set, and budget. The proxy model must expose the expected embedding/last-layer interface. | `OFFICIAL_WORKFLOW_ADAPTED` |
| GradMatch | `third_party/cords` @ `8d10c7f` | CORDS `GradMatchStrategy` computes per-sample gradients from a model and labeled train/validation loaders. | Use a CORDS-compatible ECG+PPG proxy model. This is a valid method migration, but not an original image benchmark reproduction. | `OFFICIAL_WORKFLOW_ADAPTED` |
| CRAIG | `third_party/craig_official` @ `b0374a2`; refactor at `third_party/CRAIG` @ `d276791` | Original repo is hard-wired to MNIST/CIFAR/logistic examples. Refactor uses `submodlib`; CORDS also includes CRAIG strategy. | For ECG+PPG, use CORDS or the refactored CRAIG selector with ECG+PPG gradient embeddings. Keep the original repo only for provenance and original-paper reproduction checks. | `OFFICIAL_WORKFLOW_ADAPTED` |
| TypiClust | `third_party/TypiClust` @ `4097a71` | Feature-level active learning: load embeddings, cluster unlabeled features, select typical samples. | `scripts/official_selection_adapters.py` replaces the feature loader with ECG+PPG embeddings and calls the official TypiClust class. | `OFFICIAL_CORE_ADAPTED` |
| ProbCover | `third_party/TypiClust` @ `4097a71` | Feature-level radius graph and greedy coverage selection. | `scripts/official_selection_adapters.py` replaces the feature loader with ECG+PPG embeddings and calls the official ProbCover class. | `OFFICIAL_CORE_ADAPTED` |

## ECG+PPG Adapter Contract

The common selection unit is one synchronized ECG+PPG window with a stable
`sample_id`, `record_id` or `patient_id`, split id, feature vector, and task
label. Patient or record grouping must be respected before selection and before
downstream evaluation.

Feature-based selectors receive:

```text
features: float32 [num_samples, embedding_dim]
labels: int64 [num_samples] when required by the method
budget: selected sample count
seed: reproducibility seed
```

Gradient or training-dynamics selectors receive:

```text
train_loader, val_loader
proxy_model.forward(x, last=True, freeze=True) -> logits, embedding
proxy_model.get_embedding_dim() -> int
loss: CrossEntropyLoss for classification or a deliberately audited regression loss
budget: selected sample count
```

The selector may only return sample indices. All selected subsets are then
evaluated with the same ECG+PPG downstream protocol, train proportions, seeds,
and metrics. Converting waveforms to artificial images is not part of the
default protocol because it changes the modality and invalidates direct paper
metric comparisons.

## Current Dependency Gaps

- `submodlib` is required by the CRAIG refactor.
- `apricot-select` is required by some facility-location paths in CORDS.
- PreSelect has a separate text/LLM stack and should not be installed into the
  ECG+PPG benchmark environment unless we start a text-selection experiment.

## Next Implementation Steps

1. Add a shared `SignalSelectionDataset` that wraps BIDMC and MC-MED window
   tables and exposes stable indices.
2. Add one lightweight ECG+PPG proxy model with the CORDS-compatible
   `last=True` and `get_embedding_dim()` interface.
3. Add CORDS wrappers for GradMatch, GLISTER, and CRAIG on the same BIDMC
   public protocol.
4. Keep ELFS and Moderate Coreset as feature/score adapters first, then run
   MC-MED once PhysioNet training approval becomes active.
