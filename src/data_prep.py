"""
Dataset utilities: stratified K-fold creation, test-set size balancing,
and on-disk training-set augmentation for the stroke classification pipeline.
"""
import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random
from tqdm import tqdm
from pathlib import Path

# Default class folder names; must stay aligned with `data.classes` in config.yaml.
DEFAULT_CLASSES = ["No-Stroke", "Stroke"]


def create_folds(
    data_dir: str,
    output_dir: str,
    n_splits: int = 5,
    seed: int = 42,
    classes: list | None = None,
):
    """
    Split the dataset into stratified K folds (train/test under each fold).
    """
    # Class folder names (caller-supplied or default).
    if classes is None:
        classes = list(DEFAULT_CLASSES)

    # Collect file entries and matching string labels for StratifiedKFold.
    all_files = []
    labels = []

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.exists(cls_path):
            continue
        files = os.listdir(cls_path)
        all_files.extend([(cls, f) for f in files])  # (class_name, file_name)
        labels.extend([cls] * len(files))  # same length as all_files; used as stratification labels

    # Stratified split: preserves class proportions in each fold’s train/test.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Build each fold directory and populate train/ and test/ subtrees by index.
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_files, labels)):
        fold_name = f"Fold{fold_idx + 1}"
        fold_path = os.path.join(output_dir, fold_name)
        print(f"Building [{fold_name}]...")

        # First tuple: train indices → .../train/... ; second: test indices → .../test/...
        for dataset, idx_set, subset in [("train", train_idx, "train"), ("test", test_idx, "test")]:
            for i in idx_set:
                cls, file = all_files[i]
                src = os.path.join(data_dir, cls, file)
                dst = os.path.join(fold_path, subset, cls, file)

                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)

    print("All folds created successfully.\n")


# Original augmentation stack 
# Apply multiple ops in one call


def random_augmentation(img):
    """
    Apply a random subset of spatial augmentations.
    """
    h, w = img.shape[:2]
    aug_names = []

    # Random rotation (probability 0.5); aug_names records ops for output filename suffix.
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M_rot, (w, h))
        aug_names.append(f"rot{int(angle)}")

    # Random zoom / crop back to H×W
    if random.random() < 0.5:
        zoom_factor = random.uniform(1.0, 1.2)
        img_zoom = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
        zh, zw = img_zoom.shape[:2]
        start_x = zh // 2 - h // 2
        start_y = zw // 2 - w // 2
        img = img_zoom[start_x : start_x + h, start_y : start_y + w]
        aug_names.append(f"zoom{zoom_factor:.2f}")

    # Random translation
    if random.random() < 0.5:
        tx, ty = random.randint(-10, 10), random.randint(-10, 10)
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M_trans, (w, h))
        aug_names.append(f"trans{tx}_{ty}")

    # Random horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        aug_names.append("flip")

    if not aug_names:
        aug_names.append("original")

    return img, "_".join(aug_names)


# Balance test counts per class, then grow train folders via augmentation


def balance_test_set(
    base_dir: str,
    folds: list,
    test_size_target: int = 750,
    classes: list | None = None,
):
    """
    Balance per-class test set size by moving images between train and test.
    """
    if classes is None:
        classes = list(DEFAULT_CLASSES)

    base_path = Path(base_dir)

    for fold in folds:
        for cls in classes:
            # Per-class train / test directories for this fold
            train_dir = base_path / fold / "train" / cls
            test_dir = base_path / fold / "test" / cls
            if not train_dir.exists() or not test_dir.exists():
                continue

            # List existing images in train and test
            train_files = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
            test_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))

            # Test set smaller than target: move random files from train → test
            if len(test_files) < test_size_target:
                missing = test_size_target - len(test_files)
                if missing > 0 and len(train_files) >= missing:
                    to_move = random.sample(train_files, missing)
                    for f in to_move:
                        shutil.move(str(f), str(test_dir / f.name))

            # Test set larger than target: move excess files from test → train
            elif len(test_files) > test_size_target:
                excess = len(test_files) - test_size_target
                if excess > 0:
                    to_move = random.sample(test_files, excess)
                    for f in to_move:
                        shutil.move(str(f), str(train_dir / f.name))

    print("Done. Test sets rebalanced.")


def augment_train_set(
    base_dir: str,
    folds: list,
    target_count: int = 7500,
    classes: list | None = None,
):
    """
    Augment training images until each class folder reaches target_count.
    """
    if classes is None:
        classes = list(DEFAULT_CLASSES)

    for fold in folds:
        for cls in classes:
            # Training images for this fold and class
            train_dir = os.path.join(base_dir, fold, "train", cls)
            if not os.path.exists(train_dir):
                continue

            # List images already on disk (extensions we accept for augmentation base samples)
            image_files = [
                f for f in os.listdir(train_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            current_count = len(image_files)
            needed_augments = target_count - current_count

            if needed_augments <= 0:
                print(f"{fold}/{cls}: already has {current_count} images, skipped.")
                continue

            print(f"{fold}/{cls}: generating {needed_augments} additional images.")

            i = 0

            while needed_augments > 0:
                img_name = random.choice(image_files)
                img_path = os.path.join(train_dir, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"ERROR: could not read {img_path} (missing or corrupt file).")
                    continue

                aug_img, aug_desc = random_augmentation(img)
                new_img_name = f"{os.path.splitext(img_name)[0]}_{aug_desc}_{i}.jpg"

                cv2.imwrite(os.path.join(train_dir, new_img_name), aug_img)
                needed_augments -= 1
                i += 1
