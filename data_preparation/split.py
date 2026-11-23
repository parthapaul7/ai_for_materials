import random
from collections import defaultdict
from config import SPLIT_RATIOS, RANDOM_SEED

def assign_trajectory_group_id(frame):
    """
    Determine the logical trajectory/reaction group for a frame.
    For NEB image directories (e.g. neb_1/01, neb_1/02), the parent
    directory (neb_1) is used as the group ID to ensure all images
    of a reaction coordinate remain in the same split.
    """
    src = frame.info.get('source_dir', '')
    parts = src.split('/')
    if len(parts) > 1 and parts[-1].isdigit():
        return '/'.join(parts[:-1])
    return src

def group_stratified_split(frames_by_group):
    """
    Split trajectory groups into train, val, and test sets.
    Stratifies by category to maintain balanced representation.

    frames_by_group: dict mapping group_id -> list of Atoms
    Returns: dict mapping 'train', 'val', 'test' -> list of Atoms
    """
    rng = random.Random(RANDOM_SEED)

    # Organize groups by category
    category_groups = defaultdict(list)
    for group_id, frames in frames_by_group.items():
        if not frames:
            continue
        cat = frames[0].info.get('category', 'general')
        category_groups[cat].append((group_id, frames))

    splits = {'train': [], 'val': [], 'test': []}

    for cat, groups in category_groups.items():
        # Deterministic shuffle
        rng.shuffle(groups)
        n_groups = len(groups)

        if n_groups == 1:
            # Single group (e.g. unique fragment) goes to train
            splits['train'].extend(groups[0][1])
            continue
        elif n_groups == 2:
            splits['train'].extend(groups[0][1])
            splits['val'].extend(groups[1][1])
            continue

        n_val = max(1, int(round(n_groups * SPLIT_RATIOS['val'])))
        n_test = max(1, int(round(n_groups * SPLIT_RATIOS['test'])))
        n_train = n_groups - n_val - n_test

        # Ensure at least 1 in train
        if n_train <= 0:
            n_train = 1
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1

        train_groups = groups[:n_train]
        val_groups = groups[n_train:n_train + n_val]
        test_groups = groups[n_train + n_val:]

        for _, frames in train_groups:
            splits['train'].extend(frames)
        for _, frames in val_groups:
            splits['val'].extend(frames)
        for _, frames in test_groups:
            splits['test'].extend(frames)

    # Final shuffle within each split
    rng.shuffle(splits['train'])
    rng.shuffle(splits['val'])
    rng.shuffle(splits['test'])

    return splits
