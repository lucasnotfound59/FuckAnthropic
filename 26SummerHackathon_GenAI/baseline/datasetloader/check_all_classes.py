import os
from pathlib import Path
from collections import defaultdict

label_dir = Path('dataset/bdd100k/train/labels')
class_counts = defaultdict(int)
total_files = 0

for f in os.listdir(label_dir):
    if not f.endswith('.txt'):
        continue
    total_files += 1
    classes_in_file = set()
    with open(label_dir / f) as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                classes_in_file.add(int(parts[0]))
    for c in classes_in_file:
        class_counts[c] += 1

names = {0:'person',1:'rider',2:'car',3:'truck',4:'bus',5:'train',6:'motorcycle',7:'bicycle',8:'traffic light',9:'traffic sign'}
print(f'Total label files: {total_files}')
print()
for cls in range(10):
    name = names.get(cls, '?')
    count = class_counts.get(cls, 0)
    mark = ' >=1k' if count >= 1000 else ' <1k'
    print(f'  class {cls} ({name}): {count} samples  {mark}')
