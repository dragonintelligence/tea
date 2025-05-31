# Adaptive Uncertainty and Entropy Extensions for Test-Time Energy Adaptation

This repository extends the original TEA (Test-time Energy Adaptation) framework with:
- **Uncertainty-Adaptive Temperature Scaling:** Dynamically adjusts the softmax temperature based on model uncertainty (entropy), improving calibration and robustness.
- **Entropy-Aware Energy Adaptation:** Integrates entropy-based uncertainty into the energy adaptation process.

## Features

- **Adaptive Uncertainty Model:**  
  Implements temperature scaling that adapts per input based on normalized predictive entropy.
- **Energy-based Adaptation with Entropy:**  
  Extends energy adaptation to leverage entropy for more robust test-time adaptation.
- **Support for Standard and Corrupted Datasets:**  
  Works with CIFAR-10, CIFAR-100, Tiny-ImageNet, and their corrupted variants.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd tea
   ```

2. Create the conda environment:
   ```bash
   conda env create -f tea_env.yml
   conda activate tea
   ```

3. (Optional) Install additional requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

- Place datasets in the `datasets/` directory.
- For Tiny-ImageNet-C, the structure should be:
  ```
  datasets/
    Tiny-ImageNet-C/
      gaussian_noise/
      brightness/
      ...
  ```
- For pretrained models, place checkpoints in `ckpt/` (e.g., `ckpt/tin200/resnet18_TIN.pkl`).

## Usage

### Run Adaptive Uncertainty Model

```bash
python main.py --cfg cfgs/tin200/uncertainty.yaml
```

### Run Energy Adaptation with Entropy

```bash
python main.py --cfg cfgs/tin200/energy.yaml
```

### Example SLURM Job Script

```bash
sbatch run_uncertainty.job
```

## Configuration

- All settings can be controlled via YAML config files in `cfgs/`.
- Example: `cfgs/tin200/uncertainty.yaml` for Tiny-ImageNet with uncertainty adaptation.

## Main Modules

- `core/adazoo/uncertainty.py`: Implements the UncertaintyModel and adaptive temperature scaling.
- `core/adazoo/energy.py`: Implements energy-based adaptation, extended for entropy.
- `core/data.py`: Data loading and preprocessing.
- `core/model/`: Model architectures (ResNet, WideResNet, etc.).

## Extending the Framework

- Add new adaptation methods in `core/adazoo/`.
- Register new models in `core/model/`.
- Add new configs in `cfgs/`.

## Notes for Entropy-based Energy Adaptation

- **Important:** To enable entropy-based energy adaptation, you must set `filtering: true` in your energy adaptation config (e.g., in `cfgs/tin200/energy.yaml`).

## Adaptive Calibration Parameters

The following parameters control the behavior of adaptive uncertainty and calibration:

- **temperature**: Base temperature for softmax scaling. Higher values produce softer probability distributions.
- **min_temperature**: The minimum allowed temperature during annealing or adaptation.
- **scaling_factor**: Multiplies the normalized entropy to adapt the temperature per input. Higher values make the temperature more sensitive to uncertainty.
- **uncertainty_threshold**: Threshold for considering a prediction as uncertain. Can be used for filtering or special handling of high-uncertainty samples.
- **contrast_boost**: (If used) Controls the strength of contrastive adaptation in the uncertainty model.
- **noise_boost**: (If used) Controls the amount of noise injected during adaptation for robustness.

Tune these parameters in your config files (e.g., `cfgs/tin200/uncertainty.yaml`) to optimize calibration and adaptation for your dataset and model.
