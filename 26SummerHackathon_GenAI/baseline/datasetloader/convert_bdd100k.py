import os
import glob
import json
from pathlib import Path

BDD100K_CLASSES = {
    'person': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motor': 6,
    'bike': 7,
    'traffic light': 8,
    'traffic sign': 9,
}

IMG_WIDTH = 1280
IMG_HEIGHT = 720

def convert_json_to_yolo(json_path, output_label_path):
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
    
    with open(output_label_path, 'w') as f:
        if lines:
            f.write('\n'.join(lines))

def main():
    dataset_root = Path('dataset') / 'bdd100k'
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        labels_dir = dataset_root / split / 'labels'
        
        json_files = glob.glob(str(labels_dir / '*.json'))
        converted_count = 0
        
        for json_file in json_files:
            json_path = Path(json_file)
            txt_path = labels_dir / (json_path.stem + '.txt')
            
            if not txt_path.exists():
                convert_json_to_yolo(str(json_path), str(txt_path))
                converted_count += 1
        
        total = len(json_files)
        print(f"Converted {converted_count}/{total} labels for {split}")
    
    print("\nDone! All JSON labels converted to YOLO TXT format.")

if __name__ == '__main__':
    main()
