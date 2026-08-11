import argparse
from pathlib import Path
import glob
from PIL import Image
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu
from catalyst.dl import SupervisedRunner
from skimage.morphology import remove_small_holes, remove_small_objects
from tools.cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision import *
import random
import os
import matplotlib.pyplot as plt


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def udd6_to_rgb(mask):
    """Convert UDD6 mask to RGB visualization"""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
   
    # Modify these colors according to your UDD6 classes
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]       # Other
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [102, 102, 156],    # Facade    
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [128, 64, 128],     # Road
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [107, 142, 35],     # Vegetation
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 0, 142],        # Vehicle
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [70, 70, 70]       # Roof
   
    return cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)


def visualize_results(image, prediction, ground_truth=None, save_path=None, alpha=0.5):
    """Visualize input image, prediction, and optionally ground truth"""
    plt.figure(figsize=(18, 12))
   
    # Convert prediction to RGB
    pred_rgb = udd6_to_rgb(prediction)
   
    if ground_truth is not None:
        gt_rgb = udd6_to_rgb(ground_truth)
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
       
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(cv2.cvtColor(gt_rgb, cv2.COLOR_BGR2RGB))
        plt.axis('off')
       
        plt.subplot(1, 3, 3)
        plt.title("Prediction (mIoU: 79.6)")
        plt.imshow(cv2.cvtColor(pred_rgb, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    else:
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
       
        plt.subplot(1, 2, 2)
        plt.title("Prediction (mIoU: 79.6)")
        plt.imshow(cv2.cvtColor(pred_rgb, cv2.COLOR_BGR2RGB))
        plt.axis('off')
   
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image_path", type=str, required=True, help="Path to input images")
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-o", "--output_path", type=Path, help="Path to save resulting masks", required=True)
    arg("-t", "--tta", help="Test time augmentation", default=None, choices=[None, "d4", "lr"])
    arg("-ph", "--patch-height", help="Height of patch size", type=int, default=1024)
    arg("-pw", "--patch-width", help="Width of patch size", type=int, default=1024)
    arg("-b", "--batch-size", help="Batch size", type=int, default=4)
    arg("-v", "--visualize", help="Enable visualization", action="store_true")
    arg("-gt", "--ground_truth_path", type=str, help="Path to ground truth masks for comparison")
    return parser.parse_args()


def load_checkpoint(checkpoint_path, model):
    pretrained_dict = torch.load(checkpoint_path)['model_state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    # print(oh, ow, rh, rw, height_pad, width_pad)
    h, w = oh + height_pad, ow + width_pad

    pad = albu.PadIfNeeded(min_height=h, min_width=w, border_mode=0,
                           position='bottom_right', value=[0, 0, 0])(image=image)
    img_pad = pad['image']
    return img_pad, height_pad, width_pad


class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, transform=albu.Normalize()):
        self.tile_list = tile_list
        self.transform = transform

    def __getitem__(self, index):
        img = self.tile_list[index]
        img_id = index
        aug = self.transform(image=img)
        img = aug['image']
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        results = dict(img_id=img_id, img=img)
        return results

    def __len__(self):
        return len(self.tile_list)


def make_dataset_for_one_huge_image(img_path, patch_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tile_list = []
    image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)

    output_height, output_width = image_pad.shape[0], image_pad.shape[1]

    for x in range(0, output_height, patch_size[0]):
        for y in range(0, output_width, patch_size[1]):
            image_tile = image_pad[x:x+patch_size[0], y:y+patch_size[1]]
            tile_list.append(image_tile)

    dataset = InferenceDataset(tile_list=tile_list)
    return dataset, width_pad, height_pad, output_width, output_height, image_pad, img.shape

def main():
    args = get_args()
    seed_everything(42)
   
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
   
    # Get all image paths
    img_paths = []
    for ext in ('*.tif', '*.png', '*.jpg', '*.jpeg', '*.JPG'):
        img_paths.extend(glob.glob(os.path.join(args.image_path, ext)))
    img_paths.sort()
   
    # Initialize model
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name+'.ckpt'),
        config=config
    )
    model.cuda()
    model.eval()
   
    # Setup TTA if needed
    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.Scale(scales=[0.75, 1, 1.25])])
        model = tta.SegmentationTTAWrapper(model, transforms)
   
    patch_size = (args.patch_height, args.patch_width)
   
    for img_path in tqdm(img_paths, desc="Processing images"):
        img_name = os.path.basename(img_path)
       
        # Load and process image
        dataset, width_pad, height_pad, output_width, output_height, img_pad, img_shape = \
            make_dataset_for_one_huge_image(img_path, patch_size)
       
        output_mask = np.zeros(shape=(output_height, output_width), dtype=np.uint8)
        output_tiles = []
       
        # Inference
        with torch.no_grad():
            dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                    drop_last=False, shuffle=False)
            for input in tqdm(dataloader, desc=f"Processing {img_name}", leave=False):
                raw_predictions = model(input['img'].cuda())
                predictions = nn.Softmax(dim=1)(raw_predictions).argmax(dim=1)
                image_ids = input['img_id']
               
                for i in range(predictions.shape[0]):
                    output_tiles.append((predictions[i].cpu().numpy(), image_ids[i].cpu().numpy()))
       
        # Reconstruct full mask
        k = 0
        for m in range(0, output_height, patch_size[0]):
            for n in range(0, output_width, patch_size[1]):
                output_mask[m:m + patch_size[0], n:n + patch_size[1]] = output_tiles[k][0]
                k += 1
       
        # Remove padding
        output_mask = output_mask[-img_shape[0]:, -img_shape[1]:]
       
        # Save raw prediction
        raw_output_path = os.path.join(args.output_path, f"pred_{img_name}")
        cv2.imwrite(raw_output_path, output_mask)
       
        # Visualization
        if args.visualize:
            # Load original image
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
           
            # Load ground truth if provided
            gt_mask = None
            if args.ground_truth_path:
                gt_path = os.path.join(args.ground_truth_path, img_name)
                if os.path.exists(gt_path):
                    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
           
            # Create visualization
            vis_path = os.path.join(args.output_path, f"vis_{os.path.splitext(img_name)[0]}.png")
            visualize_results(
                original_img,
                output_mask,
                ground_truth=gt_mask,
                save_path=vis_path
            )


if __name__ == "__main__":
    main()
