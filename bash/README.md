# Bash scripts

## Layout

| Script | Purpose |
|---|---|
| `env.sh` | Activates the `hogdiff` conda env and exports `CUDA_VISIBLE_DEVICES`, `CKPT_ROOT`, `DATA_ROOT`. Sourced by `sample.sh`; do not run directly. |
| `sample.sh` | Sampling entry point: `bash bash/sample.sh <config>`. |

Override defaults by exporting variables before sourcing:

```bash
CUDA_VISIBLE_DEVICES=1 CKPT_ROOT=/custom/ckpts bash bash/sample.sh qm9
```

## Sampling

```bash
bash bash/sample.sh qm9
bash bash/sample.sh zinc250k
bash bash/sample.sh cs         # community_small
bash bash/sample.sh ego        # ego_small
bash bash/sample.sh sbm
bash bash/sample.sh moses
bash bash/sample.sh guacamol
bash bash/sample.sh enzymes
```

Equivalent to:

```bash
source bash/env.sh
python ./main.py --config <config> --mode sample
```

## Training

`sample.sh` only wraps sampling. For training, source the env first:

```bash
source bash/env.sh
```

### Stage 1: higher-order diffusion

```bash
python main.py --config <config> --mode train_ho --exp_name <name>
```

### Stage 2: OU bridge

```bash
python main.py --config <config> --mode train_OU --exp_name <name>
```

### Resume training from a checkpoint

```bash
python main.py --config sbm --mode train_OU \
  --exp_name sbm-OU \
  --ckpt_meta ./checkpoints/sbm/<exp_name>/<ckpt>.pth
```

### Sample with a specific checkpoint

```bash
python main.py --config sbm --mode sample \
  --sample_prior None \
  --ckpt /path/to/ckpt.pth
```

## Key CLI flags

| Flag | Meaning |
|---|---|
| `--config` | Config name (matches `configs/<name>.yaml`, no extension). |
| `--mode` | `train_ho` / `train_OU` / `sample`. |
| `--exp_name` | Experiment name (log / checkpoint directory). |
| `--ckpt` | Checkpoint path for sampling. |
| `--ckpt_meta` | Checkpoint path for resumed training. |
| `--sample_prior` | `None` (default, two-stage) or `mu` (skip HO, use ground-truth as phase-2 prior). |
