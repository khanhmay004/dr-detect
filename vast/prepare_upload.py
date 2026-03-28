"""
Prepare and compress data for upload to Vast.ai
"""
import zipfile
import tarfile
from pathlib import Path
import hashlib
import json
from datetime import datetime

def calculate_md5(filepath):
    """Calculate MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compress_source_code(output_path="dr-detect-src.tar.gz"):
    """Compress source code for upload."""
    print(f"Compressing source code to {output_path}...")
    
    files_to_include = [
        "src/",
        "requirements.txt",
        "README.md" if Path("README.md").exists() else None
    ]
    
    with tarfile.open(output_path, "w:gz") as tar:
        for item in files_to_include:
            if item and Path(item).exists():
                print(f"  Adding {item}")
                tar.add(item)
    
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    checksum = calculate_md5(output_path)
    
    print(f"✓ Source code compressed: {size_mb:.2f} MB")
    print(f"  MD5: {checksum}")
    
    return output_path, checksum


def compress_aptos_data(data_dir="aptos/aptos2019-blindness-detection", 
                        output_path="aptos-data.tar.gz"):
    """Compress APTOS dataset for upload."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return None, None
    
    print(f"Compressing APTOS data to {output_path}...")
    print("  This may take several minutes...")
    
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(data_path, arcname="aptos2019-blindness-detection")
    
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    checksum = calculate_md5(output_path)
    
    print(f"✓ APTOS data compressed: {size_mb:.2f} MB")
    print(f"  MD5: {checksum}")
    
    return output_path, checksum


def create_manifest(files_info, output_path="upload_manifest.json"):
    """Create manifest file with checksums and metadata."""
    manifest = {
        "created_at": datetime.now().isoformat(),
        "files": files_info,
        "instructions": {
            "1_upload": "Upload all .tar.gz files to Google Drive or cloud storage",
            "2_get_links": "Get direct download links (use https://sites.google.com/site/gdocs2direct/ for Google Drive)",
            "3_vast_setup": "On Vast.ai instance, run: bash setup_vastai.sh <data_url> <src_url>",
            "4_train": "Run: bash run_training.sh"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Manifest created: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("DR-Detect Data Preparation for Vast.ai")
    print("=" * 60)
    print()
    
    files_info = {}
    
    # Compress source code
    src_path, src_checksum = compress_source_code()
    if src_path:
        files_info["source_code"] = {
            "filename": src_path,
            "checksum": src_checksum,
            "type": "source"
        }
    
    print()
    
    # Compress APTOS data
    data_path, data_checksum = compress_aptos_data()
    if data_path:
        files_info["aptos_data"] = {
            "filename": data_path,
            "checksum": data_checksum,
            "type": "dataset"
        }
    
    print()
    
    # Create manifest
    manifest_path = create_manifest(files_info)
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Upload the following files to Google Drive or cloud storage:")
    for name, info in files_info.items():
        print(f"   - {info['filename']}")
    print()
    print("2. Get direct download links for each file")
    print("   (Use https://sites.google.com/site/gdocs2direct/ for Google Drive)")
    print()
    print("3. On Vast.ai, use the setup script with your download URLs")
    print()
    print(f"4. See {manifest_path} for detailed instructions")
    print("=" * 60)


if __name__ == "__main__":
    main()
