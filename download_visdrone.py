#!/usr/bin/env python3
"""
Simplified VisDrone Dataset Download and Preparation Script
Adapted for UAV-DETR project.
"""

import gdown
import zipfile
import shutil
import json
from pathlib import Path
from PIL import Image

# Research-verified Google Drive file IDs
DATASETS = {
    "train": {
        "id": "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn",
        "filename": "VisDrone2019-DET-train.zip"
    },
    "val": {
        "id": "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59", 
        "filename": "VisDrone2019-DET-val.zip"
    }
}

# Vehicle class mappings (research-corrected)
VEHICLE_CLASSES = {4: 0, 5: 1, 6: 2, 9: 3}  # VisDrone -> COCO
CLASS_NAMES = {0: "car", 1: "van", 2: "truck", 3: "bus"}

def download_dataset(split="train"):
    """Download dataset using proven working gdown method."""
    print(f"📦 Downloading VisDrone {split} split...")
    
    # Create data directory
    data_dir = Path("data/visdrone")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download using research-verified file ID
    file_id = DATASETS[split]["id"]
    output_path = data_dir / DATASETS[split]["filename"]
    
    if output_path.exists():
        print(f"✅ File already exists: {output_path}")
        return output_path
    
    try:
        gdown.download(id=file_id, output=str(output_path), quiet=False)
        print(f"✅ {split.title()} download complete!")
        return output_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def extract_dataset(zip_path, split="train"):
    """Extract dataset using proven working zipfile method."""
    print(f"📂 Extracting {split} split...")
    
    extract_dir = Path(f"data/visdrone/extract_{split}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print(f"✅ Extraction complete to {extract_dir}")
        return extract_dir
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return None

def organize_split(extract_dir, split="train"):
    """Organize extracted files into UAV-DETR expected structure."""
    print(f"📁 Organizing {split} split for UAV-DETR...")
    
    # Find the actual data directories
    visdrone_dirs = list(extract_dir.rglob("VisDrone*"))
    if not visdrone_dirs:
        print(f"⚠️ Warning: No VisDrone directories found in {extract_dir}")
        return None
    
    source_dir = visdrone_dirs[0]
    organized_dir = Path(f"data/visdrone/organized/{split}")
    organized_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images and annotations
    for subdir in ["images", "annotations"]:
        source_subdir = source_dir / subdir
        dest_subdir = organized_dir / subdir
        
        if source_subdir.exists() and not dest_subdir.exists():
            shutil.copytree(source_subdir, dest_subdir)
            print(f"✅ Copied {subdir} to {dest_subdir}")
    
    return organized_dir

def convert_to_yolo_format(split_dir, split="train"):
    """Convert VisDrone annotations to YOLO format for UAV-DETR."""
    print(f"🏷️ Converting {split} to YOLO format...")
    
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    labels_dir = split_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    
    if not images_dir.exists() or not annotations_dir.exists():
        print(f"❌ Missing required directories in {split_dir}")
        return None
    
    image_files = sorted(images_dir.glob("*.jpg"))
    converted_count = 0
    
    print(f"📝 Processing {len(image_files)} images...")
    
    for image_file in image_files:
        # Get image dimensions
        try:
            with Image.open(image_file) as img:
                img_width, img_height = img.size
        except:
            continue
        
        # Parse annotations
        ann_file = annotations_dir / f"{image_file.stem}.txt"
        label_file = labels_dir / f"{image_file.stem}.txt"
        
        if ann_file.exists():
            yolo_annotations = []
            
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 8:
                        try:
                            x, y, w, h, score, category, trunc, occl = map(int, parts)
                            
                            # Filter: skip ignored regions and non-vehicles
                            if score == 0 or category not in VEHICLE_CLASSES:
                                continue
                            
                            # Convert to YOLO format (normalized)
                            x_center = (x + w/2) / img_width
                            y_center = (y + h/2) / img_height
                            norm_width = w / img_width
                            norm_height = h / img_height
                            
                            class_id = VEHICLE_CLASSES[category]
                            
                            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
                        except:
                            continue
            
            # Save YOLO label file
            if yolo_annotations:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations) + '\n')
                converted_count += 1
    
    print(f"✅ Converted {converted_count} images to YOLO format")
    return labels_dir

def main():
    """Main pipeline - download, extract, organize, convert for UAV-DETR."""
    print("🚀 Starting VisDrone dataset preparation for UAV-DETR...")
    
    for split in ["train", "val"]:
        print(f"\n--- Processing {split} split ---")
        
        # Step 1: Download
        zip_path = download_dataset(split)
        if not zip_path:
            continue
        
        # Step 2: Extract
        extract_dir = extract_dataset(zip_path, split)
        if not extract_dir:
            continue
        
        # Step 3: Organize
        split_dir = organize_split(extract_dir, split)
        if not split_dir:
            continue
        
        # Step 4: Convert to YOLO format
        labels_dir = convert_to_yolo_format(split_dir, split)
    
    print("\n🎯 Dataset preparation complete!")
    print("📂 Data structure ready for UAV-DETR training:")
    print("   data/visdrone/organized/train/images")
    print("   data/visdrone/organized/train/labels")
    print("   data/visdrone/organized/val/images")
    print("   data/visdrone/organized/val/labels")

if __name__ == "__main__":
    main()
