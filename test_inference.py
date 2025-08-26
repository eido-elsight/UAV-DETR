import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
from pathlib import Path

def test_model_inference():
    """
    Test the trained UAV-DETR model on a sample aerial image
    and visualize the detection results
    """
    
    # Load the trained model
    print("Loading trained UAV-DETR model...")
    model = RTDETR('runs/train/exp9/weights/best.pt')
    
    # Select a test image from validation set
    test_image_path = 'data/visdrone/yolo/images/val/0000001_02999_d_0000005.jpg'
    
    print(f"Running inference on: {test_image_path}")
    
    # Run inference
    results = model.predict(
        source=test_image_path,
        imgsz=640,
        conf=0.25,  # confidence threshold
        save=True,   # save annotated images
        project='runs/detect',
        name='test_inference'
    )
    
    # Print detection results
    print("\n=== Detection Results ===")
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            print(f"Found {len(boxes)} detections:")
            for i, box in enumerate(boxes):
                cls = int(box.cls)
                conf = float(box.conf)
                class_names = ['car', 'van', 'truck', 'bus']
                print(f"  {i+1}. {class_names[cls]}: {conf:.3f} confidence")
        else:
            print("No detections found")
    
    print(f"\nAnnotated image saved to: runs/detect/test_inference/")
    
    return results

if __name__ == "__main__":
    results = test_model_inference()
