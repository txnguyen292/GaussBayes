from pathlib import Path

file_dir = Path(__file__).resolve().parent

class CONFIG:
    data = file_dir.parent / "data"
    src = file_dir.parent / "src"
    report = file_dir.parent / "report"


if __name__ == "__main__":
    print(file_dir)
    print(CONFIG.data)