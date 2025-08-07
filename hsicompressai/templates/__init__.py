"""Configuration templates for users to customize their training setups."""

import shutil
from pathlib import Path
from typing import Dict, Any

def create_config_templates(output_dir: str, template_type: str = "basic") -> None:
    """Create configuration templates in the specified directory.
    
    Args:
        output_dir: Directory to create templates in
        template_type: Type of template (basic, advanced, custom)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    dirs = ["configs", "configs/model", "configs/data", "configs/experiment"]
    for dir_name in dirs:
        (output_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Template configurations
    templates = get_templates(template_type)
    
    # Write template files
    for file_path, content in templates.items():
        full_path = output_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    
    # Copy example data if available
    package_root = Path(__file__).parent.parent.parent
    example_configs = package_root / "configs" / "experiment"
    if example_configs.exists():
        dest = output_path / "configs" / "examples"
        dest.mkdir(exist_ok=True)
        for config_file in example_configs.glob("*.yaml"):
            shutil.copy2(config_file, dest / config_file.name)

def get_templates(template_type: str) -> Dict[str, str]:
    """Get template configurations based on type."""
    
    if template_type == "basic":
        return get_basic_templates()
    elif template_type == "advanced":
        return get_advanced_templates()
    elif template_type == "custom":
        return get_custom_templates()
    else:
        raise ValueError(f"Unknown template type: {template_type}")

def get_basic_templates() -> Dict[str, str]:
    """Basic templates for simple use cases."""
    return {
        "configs/train.yaml": """# Basic training configuration
defaults:
  - _self_
  - data: my_data
  - model: my_model
  - trainer: gpu
  - logger: wandb

# Task configuration
task_name: "my_experiment"
tags: ["basic", "custom"]

# Training settings
train: true
test: true
seed: 42

# Override default settings here
# model:
#   learning_rate: 0.001
# data:
#   batch_size: 32
""",

        "configs/data/my_data.yaml": """# Custom data configuration
# @package _global_

# Specify your data module
_target_: hsicompressai.datamodules.hyspecnet11kdatamodule.HySpecNet11kDataModule

# Data paths - update these for your dataset
data_dir: /path/to/your/data
metadata_path: /path/to/metadata.yaml

# Data loading settings
batch_size: 16
num_workers: 4
pin_memory: true

# Data splits
train_split: train
val_split: val
test_split: test

# Preprocessing
patch_size: [64, 64]
""",

        "configs/model/my_model.yaml": """# Custom model configuration
# @package _global_

# Use one of the available models or create your own
_target_: hsicompressai.models.neural.srcmodel.SRCModel

# Model architecture
input_channels: 224  # Number of spectral bands
latent_dim: 64
num_layers: 4

# Training settings
learning_rate: 0.001
weight_decay: 1e-4

# Loss function
criterion:
  _target_: hsicompressai.losses.rate_distortion.RateDistortionLoss
  lambda_rd: 0.01
""",

        "README.md": """# HSICompressAI Custom Configuration

This directory contains your custom configuration files for HSICompressAI.

## Quick Start

1. Update `configs/data/my_data.yaml` with your dataset paths
2. Modify `configs/model/my_model.yaml` with your model settings
3. Run training:
   ```bash
   hsicompressai train --config configs/train.yaml
   ```

## File Structure

- `configs/train.yaml` - Main training configuration
- `configs/data/my_data.yaml` - Dataset configuration  
- `configs/model/my_model.yaml` - Model configuration
- `configs/examples/` - Example configurations from HSICompressAI

## Customization

You can override any configuration parameter from the command line:
```bash
hsicompressai train model.learning_rate=0.0001 data.batch_size=32
```

See the HSICompressAI documentation for more advanced configurations.
"""
    }

def get_advanced_templates() -> Dict[str, str]:
    """Advanced templates with more configuration options.""" 
    basic = get_basic_templates()
    
    advanced_train = """# Advanced training configuration
defaults:
  - _self_
  - data: my_data
  - model: my_model
  - callbacks: advanced_callbacks
  - logger: multi_logger
  - trainer: gpu
  - extras: default

# Experiment tracking
experiment: my_advanced_experiment
task_name: "advanced_training"
tags: ["advanced", "production"]

# Training configuration
train: true
test: true
ckpt_path: null
seed: 42

# Hyperparameters to track
hparams:
  learning_rate: ${model.learning_rate}
  batch_size: ${data.batch_size}
  model_name: ${model._target_}
"""

    advanced_callbacks = """# Advanced callback configuration
defaults:
  - model_checkpoint
  - early_stopping
  - rich_progress_bar
  - metrics

model_checkpoint:
  monitor: "val/loss"
  mode: "min"
  save_top_k: 3
  save_last: true
  filename: "epoch_{epoch:02d}-val_loss_{val/loss:.4f}"

early_stopping:
  monitor: "val/loss"
  patience: 10
  mode: "min"
  min_delta: 0.001

learning_rate_monitor:
  _target_: pytorch_lightning.callbacks.LearningRateMonitor
  logging_interval: "step"

# Custom callbacks
reconstruction_callback:
  _target_: hsicompressai.callbacks.reconstructor.ReconstructionCallback
  save_every_n_epochs: 5
  num_samples: 4
"""

    advanced_logger = """# Multiple logger configuration
wandb:
  _target_: pytorch_lightning.loggers.wandb.WandbLogger
  project: "hsicompressai"
  name: ${task_name}
  tags: ${tags}
  save_dir: "logs/"

tensorboard:
  _target_: pytorch_lightning.loggers.tensorboard.TensorBoardLogger
  save_dir: "logs/"
  name: ${task_name}
"""

    basic.update({
        "configs/train.yaml": advanced_train,
        "configs/callbacks/advanced_callbacks.yaml": advanced_callbacks,
        "configs/logger/multi_logger.yaml": advanced_logger,
    })
    
    return basic

def get_custom_templates() -> Dict[str, str]:
    """Custom templates for specific use cases."""
    return {
        "configs/train.yaml": """# Custom training setup
# This template provides maximum flexibility for custom configurations

defaults:
  - _self_
  - data: ??? # Specify your data config
  - model: ??? # Specify your model config
  - callbacks: default
  - logger: wandb
  - trainer: gpu

# Hydra working directory
hydra:
  run:
    dir: outputs/${task_name}/${now:%Y-%m-%d_%H-%M-%S}

# Task settings
task_name: custom_experiment
tags: ["custom"]

# Training flags
train: true
test: true
ckpt_path: null

# Random seed
seed: null

# Additional configurations can be added here
""",

        "configs/data/custom_dataset.yaml": """# Template for custom dataset configuration
# @package _global_

# Custom dataset module - implement your own
_target_: your_package.datamodules.CustomDataModule

# Dataset paths
data_dir: ${paths.data_dir}
train_file: train.csv
val_file: val.csv
test_file: test.csv

# Data loading
batch_size: 16
num_workers: 4
pin_memory: true
drop_last: false

# Data preprocessing
transforms:
  train:
    - _target_: hsicompressai.transforms.normalisation.MinMaxNorm
    - _target_: torchvision.transforms.RandomHorizontalFlip
      p: 0.5
  
  val:
    - _target_: hsicompressai.transforms.normalisation.MinMaxNorm

# Dataset specific parameters
patch_size: [64, 64]
overlap: 0.1
spectral_bands: 224
""",

        "your_custom_model.py": '''"""Example custom model implementation."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig


class CustomHSIModel(pl.LightningModule):
    """Custom hyperspectral image compression model."""
    
    def __init__(
        self,
        input_channels: int = 224,
        latent_dim: int = 64,
        learning_rate: float = 1e-3,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Define your architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, latent_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(latent_dim, input_channels, 3, padding=1),
        )
        
    def forward(self, x):
        """Forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    def training_step(self, batch, batch_idx):
        """Training step.""" 
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train/loss", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val/loss", loss)
        return loss
        
    def configure_optimizers(self):
        """Configure optimizers."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
''',

        "README.md": """# Custom HSICompressAI Configuration

This template provides maximum flexibility for creating custom training setups.

## Setup

1. Implement your custom dataset in `your_package/datamodules/CustomDataModule`
2. Implement your custom model (see `your_custom_model.py` example)
3. Update config files with your specific settings
4. Run training with your custom configurations

## Custom Components

### Data Module
Create a PyTorch Lightning DataModule that follows this interface:
```python
class CustomDataModule(pl.LightningDataModule):
    def prepare_data(self): ...
    def setup(self, stage): ...
    def train_dataloader(self): ...
    def val_dataloader(self): ...
    def test_dataloader(self): ...
```

### Model
Your model should inherit from `pl.LightningModule` and implement:
- `forward()` - model forward pass
- `training_step()` - training logic
- `validation_step()` - validation logic
- `configure_optimizers()` - optimizer setup

### Running
```bash
hsicompressai train --config configs/train.yaml
```

## Advanced Usage

Override any parameter from command line:
```bash
hsicompressai train model.learning_rate=0.0001 data.batch_size=64
```

Use different configurations:
```bash
hsicompressai train data=custom_dataset model=custom_model
```
"""
    }