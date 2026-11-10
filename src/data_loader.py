from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def get_data_path(filename: str) -> Path:
    return DATA_DIR / filename
