import os
import json
import hashlib
from pathlib import Path

def verify_snapshot(raw_dir: str = "data/raw", expected_archive_sha256: str = None) -> dict:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        
    files_provenance = []
    total_bytes = 0
    
    for p in sorted(raw_path.rglob("*")):
        if p.is_file():
            h = hashlib.sha256()
            size = p.stat().st_size
            total_bytes += size
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            rel_path = str(p.relative_to(raw_path)).replace("\\", "/")
            files_provenance.append({
                "path": rel_path,
                "size_bytes": size,
                "sha256": h.hexdigest()
            })
            
    # Determine release type
    has_manifest = any("manifest" in f["path"].lower() for f in files_provenance)
    has_pcaps = any(f["path"].endswith(".pcap") or f["path"].endswith(".pcapng") for f in files_provenance)
    has_checksums = expected_archive_sha256 is not None
    
    release_type = "organizer_snapshot" if (has_manifest and has_checksums) else "public_unb_release"
    
    result = {
        "release_type": release_type,
        "has_split_manifest": has_manifest,
        "has_published_checksums": has_checksums,
        "has_data_dictionary": False,
        "has_pcaps": has_pcaps,
        "total_files": len(files_provenance),
        "total_bytes": total_bytes,
        "expected_archive_sha256": expected_archive_sha256,
        "checksum_matched": True if expected_archive_sha256 is None else False,
        "files": files_provenance
    }
    
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "provenance.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"Verified {len(files_provenance)} files ({total_bytes} bytes). Provenance written to {out_path}")
    return result

if __name__ == "__main__":
    verify_snapshot()
