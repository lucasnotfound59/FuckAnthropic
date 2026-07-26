import os
import shutil
import random
from pathlib import Path

BDD100K_ROOT = Path('/Users/yangwx/Desktop/26SummerHackathon_GenAI/baseline/dataset/bdd100k')
BDD10K_ROOT = Path('/Users/yangwx/Desktop/26SummerHackathon_GenAI/baseline/dataset/bdd10k')

NUM_TRAIN = 7000
NUM_VAL = 2000
NUM_TEST = 1000

BDD100K_CLASSES = {
    'person': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'traffic light': 8,
    'traffic sign': 9
}

def create_bdd10k():
    if BDD10K_ROOT.exists():
        shutil.rmtree(BDD10K_ROOT)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(BDD10K_ROOT / 'images' / split, exist_ok=True)
        os.makedirs(BDD10K_ROOT / 'labels' / split, exist_ok=True)
    
    split_config = {
        'train': NUM_TRAIN,
        'val': NUM_VAL,
        'test': NUM_TEST
    }
    
    for split, num_samples in split_config.items():
        img_dir = BDD100K_ROOT / split / 'images'
        label_dir = BDD100K_ROOT / split / 'labels'
        
        all_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        selected = random.sample(all_imgs, num_samples)
        
        for img_name in selected:
            img_stem = img_name[:-4]
            
            src_img = img_dir / img_name
            dst_img = BDD10K_ROOT / 'images' / split / img_name
            shutil.copy2(src_img, dst_img)
            
            txt_label = label_dir / f'{img_stem}.txt'
            if txt_label.exists():
                dst_label = BDD10K_ROOT / 'labels' / split / f'{img_stem}.txt'
                shutil.copy2(txt_label, dst_label)
        
        print(f'{split}: {len(selected)} images copied')
    
    print('BDD10K dataset created successfully!')

if __name__ == '__main__':
    create_bdd10k()
