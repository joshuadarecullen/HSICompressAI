import typer
import subprocess
from typing import List

app = typer.Typer()

@app.command()
def train(params: List[str] = typer.Argument(None)):
    """
    Call the hydra training entry point, passing any CLI args directly.
    
    Example:
    hsicompressai train experiment=mamba optimizer.lr=0.0001
    """
    cmd = ["python", "-m", "training.train"] + params
    subprocess.run(cmd)

if __name__ == "__main__":
    app()
