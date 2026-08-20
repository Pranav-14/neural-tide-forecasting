"""Package and verify final_submission.zip archive."""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch

# Keys that exist only to resume training; shipping them triples the archive size.
TRAINING_ONLY_KEYS = ("optimizer_state", "scheduler_state", "optimizer_state_dict")


def slim_checkpoint(src_path: Path, dst_path: Path) -> None:
    """Write an inference-only copy of the checkpoint, dropping training-resume state."""
    ck = torch.load(src_path, map_location="cpu")
    if not isinstance(ck, dict):
        shutil.copyfile(src_path, dst_path)
        return
    dropped = [k for k in TRAINING_ONLY_KEYS if k in ck]
    slim = {k: v for k, v in ck.items() if k not in TRAINING_ONLY_KEYS}
    torch.save(slim, dst_path)
    before, after = src_path.stat().st_size, dst_path.stat().st_size
    print(f"  * Stripped {dropped} -> {before:,} bytes reduced to {after:,} bytes")

def create_and_verify_submission():
    root = Path(__file__).resolve().parent.parent
    sub_dir = root / "submission"
    zip_path = root / "final_submission.zip"

    if zip_path.exists():
        zip_path.unlink()

    print(f"Creating {zip_path.name} from {sub_dir}...")
    with tempfile.TemporaryDirectory() as tmp:
        slim_path = Path(tmp) / "checkpoint.pt"
        ckpt_src = sub_dir / "checkpoint.pt"
        if not ckpt_src.exists():
            raise FileNotFoundError(f"Missing required submission file: {ckpt_src}")
        slim_checkpoint(ckpt_src, slim_path)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for fname in ["predict.py", "requirements.txt"]:
                src_file = sub_dir / fname
                if not src_file.exists():
                    raise FileNotFoundError(f"Missing required submission file: {src_file}")
                z.write(src_file, arcname=fname)
                print(f"  + Added: {fname} ({src_file.stat().st_size:,} bytes)")

            z.write(slim_path, arcname="checkpoint.pt")
            print(f"  + Added: checkpoint.pt ({slim_path.stat().st_size:,} bytes, inference-only)")

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
