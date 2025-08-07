import os
import sys
from pathlib import Path
from typing import List, Optional

import hydra
import typer
from omegaconf import DictConfig

app = typer.Typer(help="HSICompressAI - Hyperspectral Image Compression using AI")

def find_package_root():
    """Find the package root directory containing configs/"""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "configs").exists():
            return current
        current = current.parent
    return Path(__file__).parent

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def _train_with_hydra(cfg: DictConfig) -> None:
    """Internal function to run training with Hydra."""
    from training.train import train
    train(cfg)

@hydra.main(version_base=None, config_path="../configs", config_name="eval")  
def _eval_with_hydra(cfg: DictConfig) -> None:
    """Internal function to run evaluation with Hydra."""
    from training.eval import evaluate
    evaluate(cfg)

@app.command()
def train(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to custom config file"),
    overrides: Optional[List[str]] = typer.Argument(None, help="Hydra config overrides (e.g., model=mymodel data.batch_size=32)")
):
    """
    Train a hyperspectral compression model.
    
    Examples:
    \b
    hsicompressai train
    hsicompressai train --config my_config.yaml
    hsicompressai train model=srcmodel data=hys11-mini
    hsicompressai train experiment=mamba trainer.max_epochs=100
    """
    # Set up config path
    package_root = find_package_root()
    config_path = package_root / "configs"
    
    # Change to package directory for relative imports
    original_cwd = os.getcwd()
    os.chdir(package_root)
    
    try:
        # Build hydra command args
        args = []
        
        if config:
            # Custom config file
            config_path_full = Path(config).resolve()
            if not config_path_full.exists():
                typer.echo(f"Error: Config file not found: {config}", err=True)
                raise typer.Exit(1)
            args.extend(["--config-path", str(config_path_full.parent)])
            args.extend(["--config-name", config_path_full.stem])
        
        # Add overrides
        if overrides:
            args.extend(overrides)
            
        # Set up sys.argv for hydra
        sys.argv = ["hsicompressai"] + args
        
        _train_with_hydra()
        
    finally:
        os.chdir(original_cwd)

@app.command() 
def evaluate(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to custom config file"),
    overrides: Optional[List[str]] = typer.Argument(None, help="Hydra config overrides")
):
    """
    Evaluate a trained model.
    
    Examples:
    \b
    hsicompressai evaluate
    hsicompressai evaluate --config my_eval_config.yaml
    hsicompressai evaluate ckpt_path=path/to/checkpoint.ckpt
    """
    package_root = find_package_root()
    original_cwd = os.getcwd()
    os.chdir(package_root)
    
    try:
        args = []
        
        if config:
            config_path_full = Path(config).resolve()
            if not config_path_full.exists():
                typer.echo(f"Error: Config file not found: {config}", err=True)
                raise typer.Exit(1)
            args.extend(["--config-path", str(config_path_full.parent)])
            args.extend(["--config-name", config_path_full.stem])
        
        if overrides:
            args.extend(overrides)
            
        sys.argv = ["hsicompressai"] + args
        _eval_with_hydra()
        
    finally:
        os.chdir(original_cwd)

@app.command()
def init_config(
    output_dir: str = typer.Argument(..., help="Directory to create config templates"),
    template: str = typer.Option("basic", help="Template type: basic, advanced, custom")
):
    """
    Create config templates for training your own models.
    
    Examples:
    \b
    hsicompressai init-config my_project
    hsicompressai init-config my_project --template advanced
    """
    from .templates import create_config_templates
    create_config_templates(output_dir, template)
    typer.echo(f"Config templates created in: {output_dir}")

def main():
    """Main entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
