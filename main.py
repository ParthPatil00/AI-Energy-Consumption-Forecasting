from pathlib import Path
import sys

# Allow imports from src/
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(project_root=ROOT)
