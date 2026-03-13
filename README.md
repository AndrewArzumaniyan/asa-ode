# ASA-ODE Baseline

Baseline Neural ODE (without attention) for irregular time series interpolation on PhysioNet 2012 style data.

## Environment

Use your conda env and install PyTorch explicitly for your platform first:

```bash
conda activate py3.11_asa_ode
```


### macOS (Apple Silicon, MPS)
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Linux/Windows with NVIDIA GPU (CUDA)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### CPU-only
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

PyTorch wheel selection is not based on runtime GPU availability, so explicit install is required.
Use the official selector for the exact command: https://docs.pytorch.org/get-started/locally/

## Expected data format

The loader expects patient `.txt` files recursively under `data_root` with rows:

```text
Time,Parameter,Value
00:00,HR,80
00:00,Temp,36.9
```

`RecordID` is ignored automatically.

### Download command
```bash
curl -L -o set-a.zip https://physionet.org/files/challenge-2012/1.0.0/set-a.zip 
unzip -q set-a.zip
```
After that you will have the data in the current directory. This path is needed to be in the config. Easy way first of all:
```bash
mkdir -p data/physionet2012
cd data/physionet2012
```
and after that run commands upper to download. Paths to this directory have already added to config.

## Train

```bash
python scripts/train.py --config configs/baseline.json --rebuild-cache
```

Artifacts are saved into `output_dir` from config:
- `best_model.pt`
- `history.json`
- `summary.json`

## Evaluate

```bash
python scripts/eval.py --config configs/baseline.json --checkpoint outputs/baseline/best_model.pt
```

## Notes

- Device is selected automatically: CUDA -> MPS -> CPU.
- On MPS, adjoint is disabled automatically for stability.
- Long stages use tqdm (feature inference, parsing, stats, train/val/test loops).
