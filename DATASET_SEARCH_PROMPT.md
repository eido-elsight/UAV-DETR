# AI Research Prompt: VisDrone Dataset Acquisition for UAV-DETR

## Project Context
I am working on UAV-DETR (Unmanned Aerial Vehicle Detection and Tracking using RT-DETR), a specialized transformer-based object detection model for aerial vehicle detection. This is part of a production deployment for Elsight Ltd's Halo embedded ARM device.

## Current Status
- Successfully implemented UAV-DETR architecture with specialized aerial modules
- Docker environment ready with PyTorch 2.6.0 + CUDA 12.6 + Ubuntu 24.04
- Training pipeline configured and tested
- Need dataset to begin production training

## Dataset Requirements

### Target Dataset: VisDrone-2019-DET
- **Primary Requirement**: VisDrone-2019 Object Detection dataset
- **Format Needed**: YOLO format (or convertible to YOLO)
- **Classes Required**: Vehicle detection (car, van, truck, bus)
- **Image Count**: ~6,400+ training images, ~500+ validation images
- **Resolution**: Aerial/drone perspective imagery at 640x640 or higher

### Technical Specifications
- **Annotation Format**: YOLO format preferred (class_id x_center y_center width height)
- **File Structure**: Organized train/val splits with images/ and labels/ directories
- **Dataset Size**: Expecting ~1.5GB total download size
- **Download Method**: Direct download links, Google Drive, or automated scripts

### Model Architecture Context
- **Base Model**: RT-DETR transformer architecture
- **Specialized Modules**: UAV-specific enhancements for aerial detection
- **Training Target**: 4-class vehicle detection (car=0, van=1, truck=2, bus=3)
- **Performance Goal**: >70% mAP@0.5 for production deployment

## What I Need from You

### Primary Search Objectives
1. **Verified Download Sources**: Direct links to VisDrone-2019-DET dataset
2. **Format Specifications**: Confirm annotation format and conversion requirements
3. **Dataset Statistics**: Exact image counts, class distributions, file sizes
4. **Download Instructions**: Complete automated download process

### Specific Information Required
- **Google Drive File IDs** (if hosted on Google Drive)
- **Official repository links** (GitHub, academic sites)
- **Mirror sites** or alternative download sources
- **Checksum/verification** methods for data integrity
- **Conversion scripts** if not in YOLO format

### Docker Compatibility
- Download methods that work within Docker containers
- No interactive authentication if possible
- Automated extraction and organization scripts
- Verification that files are successfully downloaded

## Expected Deliverables

Please provide:
1. **Direct download commands** (wget, curl, or Python scripts)
2. **Complete file organization structure** after download
3. **Annotation format details** and conversion process if needed
4. **Verification steps** to ensure dataset integrity
5. **Any preprocessing requirements** for UAV-DETR compatibility

## Technical Environment
- **OS**: Ubuntu 24.04 LTS in Docker container
- **Python**: 3.12 with PyTorch 2.6.0
- **Storage**: /workspace/data/visdrone/ target directory
- **Network**: Standard internet access from container

## Success Criteria
The dataset acquisition is successful when:
- VisDrone-2019-DET downloaded and organized in YOLO format
- 4 vehicle classes properly mapped and labeled
- Train/validation splits ready for UAV-DETR training
- All files verified and accessible in Docker environment

Please provide the most reliable and current method to acquire this dataset for immediate production use.
