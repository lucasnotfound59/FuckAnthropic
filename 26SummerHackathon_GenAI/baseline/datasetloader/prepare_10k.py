import os
import glob
import random
import shutil
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

def main():
    random.seed(42)
    
    dataset_root = Path('dataset')
    src_yolo_root = dataset_root / 'bdd100k'
    dst_yolo_root = dataset_root / 'bdd100k_selected'
    
    splits = ['train', 'val', 'test']
    selected_counts = {'train': 7000, 'val': 2000, 'test': 1000}
    
    for split in splits:
        src_images_dir = src_yolo_root / split / 'images'
        src_labels_dir = src_yolo_root / split / 'labels'
        dst_images_dir = dst_yolo_root / 'images' / split
        dst_labels_dir = dst_yolo_root / 'labels' / split
        
        os.makedirs(dst_images_dir, exist_ok=True)
        os.makedirs(dst_labels_dir, exist_ok=True)
        
        img_files = sorted(glob.glob(str(src_images_dir / '*.jpg')))
        num_select = selected_counts[split]
        
        selected_files = random.sample(img_files, min(num_select, len(img_files)))
        
        for img_file in selected_files:
            src_img_path = Path(img_file)
            dst_img_path = dst_images_dir / src_img_path.name
            if not dst_img_path.exists():
                shutil.copy2(str(src_img_path), str(dst_img_path))
            
            label_name = src_img_path.stem + '.txt'
            src_label_path = src_labels_dir / label_name
            dst_label_path = dst_labels_dir / label_name
            if src_label_path.exists() and not dst_label_path.exists():
                shutil.copy2(str(src_label_path), str(dst_label_path))
        
        print(f"Selected {len(selected_files)} images and labels for {split}")
    
    total = sum(selected_counts.values())
    print(f"\nTotal selected: {total} images")

if __name__ == '__main__':
    main()
