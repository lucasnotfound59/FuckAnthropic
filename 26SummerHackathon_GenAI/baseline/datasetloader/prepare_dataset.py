import os
import glob
import shutil
from pathlib import Path

def main():
    dataset_root = Path('dataset')
    
    bdd_images_root = dataset_root / 'bdd100k_images_100k' / '100k'
    bdd_labels_root = dataset_root / 'bdd100k_labels'
    
    yolo_root = dataset_root / 'bdd100k'
    yolo_images_root = yolo_root / 'images'
    yolo_labels_root = yolo_root / 'labels'
    
    for split in ['train', 'val', 'test']:
        src_images_dir = bdd_images_root / split
        dst_images_dir = yolo_images_root / split
        dst_labels_dir = yolo_labels_root / split
        
        os.makedirs(dst_images_dir, exist_ok=True)
        os.makedirs(dst_labels_dir, exist_ok=True)
        
        img_files = glob.glob(str(src_images_dir / '*.jpg'))
        for img_file in img_files:
            src_path = Path(img_file)
            dst_path = dst_images_dir / src_path.name
            if not dst_path.exists():
                shutil.copy2(str(src_path), str(dst_path))
        
        json_files = glob.glob(str(bdd_labels_root / split / '*.json'))
        for json_file in json_files:
            json_path = Path(json_file)
            label_path = dst_labels_dir / (json_path.stem + '.txt')
            if not label_path.exists():
                convert_json_to_yolo(str(json_path), str(label_path))
        
        print(f"Prepared {len(img_files)} images and {len(json_files)} labels for {split}")

BDD100K_CLASSES = {
    'bus': 0,
    'bike': 1,
    'car': 2,
    'motor': 3,
    'person': 4,
    'rider': 5,
    'traffic light': 6,
    'traffic sign': 7,
    'train': 8,
    'truck': 9,
}

IMG_WIDTH = 1280
IMG_HEIGHT = 720

def convert_json_to_yolo(json_path, output_label_path):
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    lines = []
    for frame in data.get('frames', []):
        for obj in frame.get('objects', []):
            category = obj.get('category')
            if category not in BDD100K_CLASSES:
                continue
            
            box2d = obj.get('box2d')
            if box2d is None:
                continue
            
            x1 = box2d.get('x1', 0)
            y1 = box2d.get('y1', 0)
            x2 = box2d.get('x2', 0)
            y2 = box2d.get('y2', 0)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            class_id = BDD100K_CLASSES[category]
            
            x_center = (x1 + x2) / 2 / IMG_WIDTH
            y_center = (y1 + y2) / 2 / IMG_HEIGHT
            width = (x2 - x1) / IMG_WIDTH
            height = (y2 - y1) / IMG_HEIGHT
            
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    if not lines:
        return
    
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(lines))

if __name__ == '__main__':
    main()
