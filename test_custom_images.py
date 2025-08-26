import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import os
from pathlib import Path

def test_custom_images():
    """
    Test the trained UAV-DETR model on custom aerial images
    and save annotated results for each image
    """
    
    # Load the trained model
    print("Loading trained UAV-DETR model...")
    model = RTDETR('runs/train/exp9/weights/best.pt')
    
    # Custom test images
    test_images = [
        'test_images/overhead-cars.jpg',
        'test_images/overhead.jpg', 
        'test_images/overhead-persons.jpg'
    ]
    
    class_names = ['car', 'van', 'truck', 'bus']
    
    print(f"\n=== Testing UAV-DETR on {len(test_images)} custom aerial images ===\n")
    
    for i, image_path in enumerate(test_images, 1):
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            continue
            
        print(f"🔍 [{i}/{len(test_images)}] Processing: {image_path}")
        
        # Extract image name for output folder
        image_name = Path(image_path).stem
        
        # Run inference with moderate confidence threshold
        results = model.predict(
            source=image_path,
            imgsz=640,
            conf=0.25,  # confidence threshold
            save=True,   # save annotated images
            project='runs/detect',
            name=f'custom_{image_name}'
        )
        
        # Analyze results
        total_detections = 0
        class_counts = {name: 0 for name in class_names}
        high_conf_detections = 0
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                total_detections = len(boxes)
                for box in boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    class_counts[class_names[cls]] += 1
                    if conf > 0.5:  # High confidence detections
                        high_conf_detections += 1
        
        # Print results for this image
        print(f"  📊 Results:")
        print(f"    Total detections: {total_detections}")
        print(f"    High confidence (>0.5): {high_conf_detections}")
        
        if total_detections > 0:
            print(f"    Detected vehicles:")
            for class_name, count in class_counts.items():
                if count > 0:
                    print(f"      {class_name}: {count}")
        else:
            print(f"    ❌ No vehicles detected")
        
        print(f"  💾 Annotated image saved to: runs/detect/custom_{image_name}/")
        print()
    
    print("✅ Inference completed on all custom images!")
    print("🔍 Check the annotated results in the runs/detect/ folders to see the visual detections.")

if __name__ == "__main__":
    test_custom_images()
