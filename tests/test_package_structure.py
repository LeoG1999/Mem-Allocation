import os
import subprocess
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def package_files():
    if (ROOT / ".git").exists():
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
        return [ROOT / os.fsdecode(name) for name in names if name]
    return [p for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts]


def test_package_has_no_experiment_data():
    forbidden = {".json", ".jsonl", ".csv", ".parquet", ".pdf", ".tex", ".zip"}
    for path in package_files():
        assert path.relative_to(ROOT).parts[0] not in {"data", "release_artifacts"}
        assert path.suffix not in forbidden
        if path.suffix in {".py", ".sh", ".md", ".yaml"}:
            assert not re.search(r"/(?:home|Users)/", path.read_text(encoding="utf-8"))


def test_all_numeric_tables_have_an_entrypoint():
    mapping = (ROOT / "PAPER_EXPERIMENT_MAP.md").read_text(encoding="utf-8")
    for number in range(1, 24):
        assert re.search(rf"\bTables?\b[^\n|]*\b{number}\b", mapping)


def test_required_entrypoints_exist():
    reproduce = ROOT / "scripts/reproduce"
    for number in range(1, 9):
        assert list(reproduce.glob(f"{number:02d}_*.sh"))
    assert (reproduce / "run_all.sh").exists()
