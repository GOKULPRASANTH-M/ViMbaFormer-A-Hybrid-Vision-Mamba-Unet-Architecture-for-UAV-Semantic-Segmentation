import glob
import os
import numpy as np
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
import random
from pathlib import Path
from PIL import Image

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# SDD Dataset Color Mapping (22 classes)
paved_area = np.array([128, 64, 128])  # label 1
dirt = np.array([130, 76, 0])  # label 2
grass= np.array([0, 102, 0])  # label 3
gravel=np.array([112, 103, 87])  # label 4
water=np.array([28, 42, 168])  # label 5
rocks=np.array([48, 41, 30])  # label 6
pool=np.array([0, 50, 89])  # label 7
vegetation=np.array([107, 142, 35])  # label 8
roof=np.array([70, 70, 70])  # label 9
wall=np.array([102, 102, 156])  # label 10
window=np.array([254,228,12])  # label 11
door=np.array([254,148,12])  # label 12
fence=np.array([190,153,153])  # label 13
fence_pole=np.array([153,153,153])  # label 14
person=np.array([255,22,96])  # label 15
dog=np.array([102,51,0])  # label 16
car=np.array([9,143,150])  # label 17
bicycle=np.array([119,11,32])  # label 18
tree=np.array([51,51,0])  # label 19
bald_tree=np.array([190,250,190])  # label 20
ar_maker=np.array([112,150,146])  # label 21
obstacle=np.array([2,135,115])  # label 22

num_classes = 22

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/sdd/sdd_train_val")
    parser.add_argument("--output-img-dir", default="data/sdd/train_val/images")
    parser.add_argument("--output-mask-dir", default="data/sdd/train_val/masks")
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--split-size-h", type=int, default=1024)
    parser.add_argument("--split-size-w", type=int, default=1024)
    parser.add_argument("--pad-height", type=int, default=4096)
    parser.add_argument("--pad-width", type=int, default=6144)
    return parser.parse_args()

def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    
    mask_rgb[np.all(mask_convert == 0, axis=0)] = paved_area
    mask_rgb[np.all(mask_convert == 1, axis=0)] = dirt
    mask_rgb[np.all(mask_convert == 2, axis=0)] = grass
    mask_rgb[np.all(mask_convert == 3, axis=0)] = gravel
    mask_rgb[np.all(mask_convert == 4, axis=0)] = water
    mask_rgb[np.all(mask_convert == 5, axis=0)] = rocks
    mask_rgb[np.all(mask_convert == 6, axis=0)] = pool
    mask_rgb[np.all(mask_convert == 7, axis=0)] = vegetation
    mask_rgb[np.all(mask_convert == 8, axis=0)] = roof
    mask_rgb[np.all(mask_convert == 9, axis=0)] = wall
    mask_rgb[np.all(mask_convert == 10, axis=0)] = window
    mask_rgb[np.all(mask_convert == 11, axis=0)] = door
    mask_rgb[np.all(mask_convert == 12, axis=0)] = fence
    mask_rgb[np.all(mask_convert == 13, axis=0)] = fence_pole
    mask_rgb[np.all(mask_convert == 14, axis=0)] = person
    mask_rgb[np.all(mask_convert == 15, axis=0)] = dog
    mask_rgb[np.all(mask_convert == 16, axis=0)] = car
    mask_rgb[np.all(mask_convert == 17, axis=0)] = bicycle
    mask_rgb[np.all(mask_convert == 18, axis=0)] = tree
    mask_rgb[np.all(mask_convert == 19, axis=0)] = bald_tree
    mask_rgb[np.all(mask_convert == 20, axis=0)] = ar_maker
    mask_rgb[np.all(mask_convert == 21, axis=0)] = obstacle
    
    
    return mask_rgb

def rgb2label(label):
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    label_seg[np.all(label == paved_area, axis=-1)] = 0
    label_seg[np.all(label == dirt, axis=-1)] = 1
    label_seg[np.all(label == grass, axis=-1)] = 2
    label_seg[np.all(label == gravel, axis=-1)] = 3
    label_seg[np.all(label == water, axis=-1)] = 4
    label_seg[np.all(label == rocks, axis=-1)] = 5
    label_seg[np.all(label == pool, axis=-1)] = 6
    label_seg[np.all(label == vegetation, axis=-1)] = 7
    label_seg[np.all(label == roof, axis=-1)] = 8
    label_seg[np.all(label == wall, axis=-1)] = 9
    label_seg[np.all(label == window, axis=-1)] = 10
    label_seg[np.all(label == door, axis=-1)] = 11
    label_seg[np.all(label == fence, axis=-1)] = 12
    label_seg[np.all(label == fence_pole, axis=-1)] = 13
    label_seg[np.all(label == person, axis=-1)] = 14
    label_seg[np.all(label == dog, axis=-1)] = 15
    label_seg[np.all(label == car, axis=-1)] = 16
    label_seg[np.all(label == bicycle, axis=-1)] = 17
    label_seg[np.all(label == tree, axis=-1)] = 18
    label_seg[np.all(label == bald_tree, axis=-1)] = 19
    label_seg[np.all(label == ar_maker, axis=-1)] = 20
    label_seg[np.all(label == obstacle, axis=-1)] = 21

    
    return label_seg

def pad_if_needed(image, mask, pad_height=4096, pad_width=6144):
    """
    Pads image and mask to the specified dimensions if they are smaller
    """
    pad = albu.PadIfNeeded(
        min_height=pad_height, 
        min_width=pad_width, 
        position='bottom_right',
        border_mode=0, 
        value=[0, 0, 0], 
        mask_value=[255, 255, 255]
    )(image=image, mask=mask)
    return pad['image'], pad['mask']

def split_into_patches(img, mask, split_size=(1024, 1024)):
    # Calculate original patch dimensions (before resizing)
    patch_width = img.shape[1] // 6 
    patch_height = img.shape[0] // 4  
    
    patches = []
    k = 0
    for y in range(0, img.shape[0], patch_height):
        for x in range(0, img.shape[1], patch_width):
            img_tile = img[y:y + patch_height, x:x + patch_width]
            mask_tile = mask[y:y + patch_height, x:x + patch_width]
            
            if img_tile.shape[0] == patch_height and img_tile.shape[1] == patch_width:
                # Resize to target size
                img_tile = cv2.resize(img_tile, split_size)
                mask_tile = cv2.resize(mask_tile, split_size, interpolation=cv2.INTER_NEAREST)
                patches.append((img_tile, mask_tile, k))
                k += 1
    return patches

def process_single_image(args):
    img_path, mask_path, output_img_dir, output_mask_dir, mode, split_size, pad_height, pad_width = args
    
    # Load and convert images
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    # Get image ID
    img_id = Path(img_path).stem
    
    # Pad if needed
    img_pad, mask_pad = pad_if_needed(img, mask, pad_height, pad_width)
    
    # Convert mask to label indices if in train mode
    if mode == 'train':
        mask_pad = rgb2label(mask_pad)
    
    # Split into patches
    patches = split_into_patches(img_pad, mask_pad, split_size)
    
    # Save patches
    for img_patch, mask_patch, k in patches:
        if mode == 'train':
            # For training, masks are saved as single-channel label indices
            out_mask_path = os.path.join(output_mask_dir, f"{img_id}_{k}.png")
            cv2.imwrite(out_mask_path, mask_patch)
        else:
            # For validation/test, masks can be saved as RGB
            out_mask_path = os.path.join(output_mask_dir, f"{img_id}_{k}.png")
            mask_rgb = label2rgb(mask_patch) if mode == 'train' else mask_patch
            cv2.imwrite(out_mask_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
        
        # Save image patch
        out_img_path = os.path.join(output_img_dir, f"{img_id}_{k}.png")
        cv2.imwrite(out_img_path, cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_img_dir, exist_ok=True)
    os.makedirs(args.output_mask_dir, exist_ok=True)
    
    # Get all image and mask paths
    img_paths = sorted(glob.glob(os.path.join(args.input_dir, "Images", "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(args.input_dir, "Labels", "*.png")))
    
    # Prepare arguments for parallel processing
    process_args = [
        (img_path, mask_path, args.output_img_dir, args.output_mask_dir, args.mode,
         (args.split_size_h, args.split_size_w), args.pad_height, args.pad_width)
        for img_path, mask_path in zip(img_paths, mask_paths)
    ]
    
    # Process images in parallel
    t0 = time.time()
    with mpp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_single_image, process_args)
    t1 = time.time()
    
    print(f'Image processing completed in {t1-t0:.2f} seconds')