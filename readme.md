
# CNN vs Quaternion CNN — PathMNIST

Educational research benchmark, **not an ataxia classifier or clinical diagnostic tool**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python experiment.py --epochs 8 --batch-size 64 --max-train 12000
```

Remove `--max-train` to use the complete training split. Use `--max-val` and `--max-test` for faster experiments. First execution downloads PathMNIST automatically. Results are written to `results/`.

## Notes

- Both networks use identical 4-channel inputs: RGB + deterministic grayscale derived from RGB. The fourth channel is **not an independent imaging modality**.
- Both networks classify nine PathMNIST classes.
- The official train/validation/test partitions remain separate. Optional subset limits select deterministic random subsets within each partition.
- Validation selects the best checkpoint for each model; test metrics are computed only after training.
- Reported test metrics from a single run do not establish superiority. Repeat with multiple seeds for research claims.
- The QuaternionConv2d layer implements a Hamilton-product convolution with four shared real kernels.
