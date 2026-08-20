"""Package and verify final_submission.zip archive."""

import os
import shutil
import zipfile
from pathlib import Path

def create_and_verify_submission():
    root = Path(__file__).resolve().parent.parent
    sub_dir = root / "submission"
    zip_path = root / "final_submission.zip"

    if zip_path.exists():
        zip_path.unlink()

    print(f"Creating {zip_path.name} from {sub_dir}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        # Add root files
        for fname in ["predict.py", "requirements.txt", "checkpoint.pt"]:
            src_file = sub_dir / fname
            if not src_file.exists():
                raise FileNotFoundError(f"Missing required submission file: {src_file}")
            z.write(src_file, arcname=fname)
            print(f"  + Added: {fname} ({src_file.stat().st_size:,} bytes)")

        # Add src/model.py
        src_model = sub_dir / "src" / "model.py"
        if not src_model.exists():
            raise FileNotFoundError(f"Missing model file: {src_model}")
        z.write(src_model, arcname="src/model.py")
        print(f"  + Added: src/model.py ({src_model.stat().st_size:,} bytes)")

    print("\n=== VERIFYING ARCHIVE STRUCTURE ===")
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            print(f"  {info.filename:<20} | {info.file_size:>10,} bytes")

    print("\n[SUCCESS] final_submission.zip created and verified successfully!")

if __name__ == "__main__":
    create_and_verify_submission()
