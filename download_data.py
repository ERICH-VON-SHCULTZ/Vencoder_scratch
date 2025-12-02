#!/usr/bin/env python3
"""
fast_download.py
Download dataset from HuggingFace in parts and extract
"""

import os
import zipfile
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def download_and_extract_dataset(output_dir="./data", cache_dir="./hf_cache"):
    """
    Download the dataset parts and extract them.
    Much faster than using load_dataset!
    """
    print("="*60)
    print("Fast Download: HuggingFace Dataset Parts")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    repo_id = "tsbpp/fall2025_deeplearning"
    
    # The 5 parts
    parts = [
        "cc3m_96px_part1.zip",
        "cc3m_96px_part2.zip",
        "cc3m_96px_part3.zip",
        "cc3m_96px_part4.zip",
        "cc3m_96px_part5.zip"
    ]
    
    print(f"\nDownloading {len(parts)} parts (~2.6 GB total)...")
    print("This should take 10-60 minutes depending on your connection\n")
    
    extracted_dirs = []
    
    for i, part in enumerate(parts, 1):
        print(f"\n[{i}/{len(parts)}] Downloading {part}...")
        
        try:
            # Download the zip file
            zip_path = hf_hub_download(
                repo_id=repo_id,
                filename=part,
                repo_type="dataset",
                cache_dir=cache_dir
            )
            
            print(f"✓ Downloaded to: {zip_path}")
            
            # Extract
            extract_dir = os.path.join(output_dir, f"part{i}")
            print(f"  Extracting to: {extract_dir}...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"✓ Extracted!")
            extracted_dirs.append(extract_dir)
            
        except Exception as e:
            print(f"✗ Error with {part}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Download Complete!")
    print(f"{'='*60}")
    print(f"Extracted to: {output_dir}/")
    print(f"Parts downloaded: {len(extracted_dirs)}/{len(parts)}")
    
    return extracted_dirs



if __name__ == "__main__":
    # Download and extract
    extracted = download_and_extract_dataset()