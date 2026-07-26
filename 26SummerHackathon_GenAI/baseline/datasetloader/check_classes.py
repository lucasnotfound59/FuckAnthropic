import os
from pathlib import Path
from collections import defaultdict

label_dir = Path('dataset/bdd100k/train/labels')
class_counts = defaultdict(int)
total_valid = 0

for f in os.listdir(label_dir):
    if not f.endswith('.txt'):
        continue
    classes_in_file = set()
    has_excluded = False
    with open(label_dir / f) as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                if cls in {1,3,5,8}:
                    has_excluded = True
                    break
                classes_in_file.add(cls)
    if has_excluded:
        continue
    total_valid += 1
    for c in classes_in_file:
        class_counts[c] += 1

names = {0:'person',2:'car',4:'bus',6:'motorcycle',7:'bicycle',9:'traffic sign'}
print(f'Total valid train images (no excluded classes): {total_valid}')
for cls in sorted(class_counts):
    print(f'  class {cls} ({names.get(cls,"?")}): {class_counts[cls]} samples')
