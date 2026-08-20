"""Unit test for testing the final_submission.zip package extraction and execution."""

import subprocess
import tempfile
import zipfile
from pathlib import Path
import pandas as pd


def test_submission_zip_package():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "benchmark"
    zip_path = root / "final_submission.zip"

    assert zip_path.exists(), "final_submission.zip does not exist!"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        print(f"Testing sandbox extraction to: {tmp_path}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp_path)

        # Verify required files
        for req_file in ["predict.py", "requirements.txt", "checkpoint.pt"]:
            assert (tmp_path / req_file).exists(), f"Missing required file: {req_file}"
        assert (tmp_path / "src" / "model.py").exists(), "Missing src/model.py"

        out_file = tmp_path / "predictions.csv"
        cmd = [
            "python",
            str(tmp_path / "predict.py"),
            "--input_dir",
            str(data_dir),
            "--output_file",
            str(out_file),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
        ]

        print("Executing inference command in sandbox...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", res.stdout)
        if res.stderr:
            print("STDERR:", res.stderr)

        assert res.returncode == 0, f"predict.py failed with returncode {res.returncode}"
        assert out_file.exists(), "Output prediction CSV was not generated!"

        # Validate prediction CSV structure
        df = pd.read_csv(out_file)
        assert list(df.columns) == ["series_id", "timestamp", "prediction"], f"Invalid columns: {df.columns}"
        assert len(df) == 32256, f"Expected 32,256 rows, got {len(df)}"
        assert not df["prediction"].isna().any(), "Found NaNs in predictions!"
        print(f"[SUCCESS] Sandbox evaluation test passed! (Generated {len(df):,} valid rows, 0 NaNs)")


if __name__ == "__main__":
    test_submission_zip_package()
