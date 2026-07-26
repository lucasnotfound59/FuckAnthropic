import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

BDD100K_ROOT = Path('/Users/yangwx/Desktop/26SummerHackathon_GenAI/baseline/dataset/bdd100k')
BDD10K_SELECTED_ROOT = Path('/Users/yangwx/Desktop/26SummerHackathon_GenAI/baseline/dataset/bdd100k_selected')

NUM_TRAIN = 7000
NUM_VAL = 2000
NUM_TEST = 1000

# Only exclude class 5 (train) - only 105 samples in full dataset
EXCLUDED_CLASSES = {5}

# Remap original class IDs to contiguous 0-8
# 0:person -> 0, 1:rider -> 1, 2:car -> 2, 3:truck -> 3, 4:bus -> 4
# 6:motorcycle -> 5, 7:bicycle -> 6, 8:traffic light -> 7, 9:traffic sign -> 8
CLASS_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 9: 8}

NEW_CLASS_NAMES = {
    0: 'person', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus',
    5: 'motorcycle', 6: 'bicycle', 7: 'traffic light', 8: 'traffic sign',
}


def scan_labels(label_dir):
    """Scan all label files, return dict: stem -> (set of original class ids, list of remapped lines)."""
    result = {}
    for f in os.listdir(label_dir):
        if not f.endswith('.txt'):
            continue
        stem = f[:-4]
        classes = set()
        has_excluded = False
        remapped_lines = []
        with open(label_dir / f, 'r') as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls in EXCLUDED_CLASSES:
                    has_excluded = True
                    break
                classes.add(cls)
                new_cls = CLASS_REMAP[cls]
                remapped_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")
        if has_excluded:
            continue
        if not classes:
            continue
        result[stem] = (classes, remapped_lines)
    return result


def select_train(img_classes, num_target):
    """Greedy selection: prioritize rare classes to ensure each reaches 1000."""
    class_counts = defaultdict(int)

    # Priority by scarcity in full dataset: motorcycle > bicycle > bus > rider > truck > person > traffic light > traffic sign > car
    priority = {6: 0, 7: 1, 4: 2, 1: 3, 3: 4, 0: 5, 8: 6, 9: 7, 2: 8}

    def rarity_score(stem):
        classes = img_classes[stem][0]
        return min((priority.get(c, 99) for c in classes), default=99)

    sorted_stems = sorted(img_classes.keys(), key=rarity_score)

    selected = []
    selected_set = set()

    # Phase 1: select all images with motorcycle or bicycle (rarest)
    for stem in sorted_stems:
        classes = img_classes[stem][0]
        if 6 in classes or 7 in classes:
            selected.append(stem)
            selected_set.add(stem)
            for c in classes:
                class_counts[c] += 1

    # Phase 2: select images for each class until it reaches 1000
    for target_cls in [4, 1, 3, 0, 8, 9]:
        for stem in sorted_stems:
            if stem in selected_set:
                continue
            if class_counts[target_cls] >= 1000:
                break
            classes = img_classes[stem][0]
            if target_cls in classes:
                selected.append(stem)
                selected_set.add(stem)
                for c in classes:
                    class_counts[c] += 1

    # Phase 3: fill remaining slots randomly
    remaining = [s for s in sorted_stems if s not in selected_set]
    random.shuffle(remaining)

    for stem in remaining:
        if len(selected) >= num_target:
            break
        selected.append(stem)
        selected_set.add(stem)
        for c in img_classes[stem][0]:
            class_counts[c] += 1

    return selected, dict(class_counts)


def select_val_test(img_classes, num_target):
    """Greedy selection for val/test: prioritize rare classes to maximize coverage."""
    class_counts = defaultdict(int)

    priority = {6: 0, 7: 1, 4: 2, 1: 3, 3: 4, 0: 5, 8: 6, 9: 7, 2: 8}

    def rarity_score(stem):
        classes = img_classes[stem][0]
        return min((priority.get(c, 99) for c in classes), default=99)

    sorted_stems = sorted(img_classes.keys(), key=rarity_score)
    random.shuffle(sorted_stems)
    sorted_stems = sorted(sorted_stems, key=rarity_score)

    selected = []
    selected_set = set()
    budget = num_target

    # Phase 1: select images with motorcycle or bicycle (up to 15% of budget)
    phase1_limit = int(num_target * 0.15)
    phase1_count = 0
    for stem in sorted_stems:
        if phase1_count >= phase1_limit:
            break
        classes = img_classes[stem][0]
        if 6 in classes or 7 in classes:
            selected.append(stem)
            selected_set.add(stem)
            for c in classes:
                class_counts[c] += 1
            phase1_count += 1

    # Phase 2: select images for each class until it reaches num_target * 0.3
    threshold = int(num_target * 0.3)
    for target_cls in [4, 1, 3, 0, 8, 9]:
        if len(selected) >= budget:
            break
        for stem in sorted_stems:
            if len(selected) >= budget:
                break
            if stem in selected_set:
                continue
            if class_counts[target_cls] >= threshold:
                break
            classes = img_classes[stem][0]
            if target_cls in classes:
                selected.append(stem)
                selected_set.add(stem)
                for c in classes:
                    class_counts[c] += 1

    # Phase 3: fill remaining slots randomly
    remaining = [s for s in sorted_stems if s not in selected_set]
    random.shuffle(remaining)

    for stem in remaining:
        if len(selected) >= num_target:
            break
        selected.append(stem)
        selected_set.add(stem)
        for c in img_classes[stem][0]:
            class_counts[c] += 1

    return selected, dict(class_counts)


def create_bdd10k_selected():
    if BDD10K_SELECTED_ROOT.exists():
        shutil.rmtree(BDD10K_SELECTED_ROOT)

    for split in ['train', 'val', 'test']:
        os.makedirs(BDD10K_SELECTED_ROOT / 'images' / split, exist_ok=True)
        os.makedirs(BDD10K_SELECTED_ROOT / 'labels' / split, exist_ok=True)

    split_config = {
        'train': (NUM_TRAIN, select_train),
        'val': (NUM_VAL, select_val_test),
        'test': (NUM_TEST, select_val_test),
    }

    for split, (num_target, select_fn) in split_config.items():
        img_dir = BDD100K_ROOT / split / 'images'
        label_dir = BDD100K_ROOT / split / 'labels'

        print(f'\nScanning {split} labels...')
        img_classes = scan_labels(label_dir)
        print(f'  {len(img_classes)} valid images found')

        selected, class_counts = select_fn(img_classes, num_target)

        print(f'  Selected {len(selected)} images')
        for orig_cls in sorted(class_counts):
            new_cls = CLASS_REMAP[orig_cls]
            name = NEW_CLASS_NAMES[new_cls]
            print(f'    class {new_cls} ({name}): {class_counts[orig_cls]} samples')

        for stem in selected:
            src_img = img_dir / f'{stem}.jpg'
            dst_img = BDD10K_SELECTED_ROOT / 'images' / split / f'{stem}.jpg'
            shutil.copy2(src_img, dst_img)

            # Write remapped labels
            remapped_lines = img_classes[stem][1]
            dst_label = BDD10K_SELECTED_ROOT / 'labels' / split / f'{stem}.txt'
            with open(dst_label, 'w') as f:
                f.writelines(remapped_lines)

    print('\nBDD10K Selected dataset created successfully!')


if __name__ == '__main__':
    create_bdd10k_selected()
