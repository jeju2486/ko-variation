"""Optional importer for historical benchmark caches; not used by fresh discovery."""
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import scipy

from .discovery_store import digest


def import_benchmark_nulls(db, work, info, root, prior):
    old_preparation = Path(prior["settings"]["benchmark_dir"])/"preparation.npz"
    if not old_preparation.exists():
        raise ValueError(f"Previous benchmark preparation is required for safe reuse: {old_preparation}")
    from .null_identity import null_source_hash, LEGACY_GLMM_HASHES, LEGACY_NULL_SOURCE
    current = null_source_hash()
    source = prior.get("null_model_source_sha256")
    if source != current:
        hashes = [r.get("sha256") for r in prior.get("inputs",[]) if r.get("name")=="glmm.py"]
        # Some legacy identities store the scorer hash separately.
        hashes += [prior.get("glmm_source_sha256"), prior.get("glmm_sha256"),prior.get("code",{}).get("glmm.py")]
        if not (current == LEGACY_NULL_SOURCE and any(h in LEGACY_GLMM_HASHES for h in hashes)):
            raise ValueError("Prior benchmark null-fitting code is not verified compatible")
    versions = prior.get("versions",prior)
    if versions.get("numpy") != np.__version__ or versions.get("scipy") != scipy.__version__:
        raise ValueError("Prior benchmark NumPy/SciPy versions differ")
    if prior["settings"].get("null_max_iter") != 100 or prior["settings"].get("null_tolerance") != 1e-7:
        raise ValueError("Prior null fitting settings differ")
    expected = [r["sha256"] for r in prior["inputs"] if r["name"] == "preparation.npz"]
    if expected != [digest(old_preparation)]:
        raise ValueError("Previous benchmark preparation changed")
    extracted = work/"import_preparation"
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(old_preparation) as archive:
        for name in ("samples.npy", "K.npy", "packed.npy"):
            with archive.open(name) as source_file, (extracted/name).open("wb") as target:
                shutil.copyfileobj(source_file,target,1<<20)
    if (list(np.load(extracted/"samples.npy", mmap_mode="r")) != info["samples"] or
            not np.array_equal(np.load(extracted/"K.npy", mmap_mode="r"), np.load(work/"K.npy", mmap_mode="r"))):
        raise ValueError("Previous sample order/covariance differs; reuse rejected")
    packed = np.load(extracted/"packed.npy", mmap_mode="r")
    for meta in (root/"null_models").glob("null_*.json"):
        old_pattern = json.loads(meta.read_text())["pattern"]
        row = db.execute("SELECT id FROM patterns WHERE bits=?", (packed[old_pattern].tobytes(),)).fetchone()
        if row:
            with db:
                db.execute("INSERT OR IGNORE INTO external_nulls VALUES (?,?)", (row[0],str(meta)))
    del packed
    for path in extracted.iterdir():
        path.unlink()
    extracted.rmdir()
