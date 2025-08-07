# HSICompressAI Custom Configuration

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
