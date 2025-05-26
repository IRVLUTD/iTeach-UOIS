# ----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
# ----------------------------------------------------------------------------------------------------

"""
This script is used to propagate masks across video frames using a bounding box prompt.
It uses bounding boxes on the first frame of a video and then uses these
bounding boxes to track and propagate the masks across the subsequent frames.
"""

import json
import os
import cv2
import torch
import requests
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image as PILImg
from absl import app, flags, logging
from robokit.utils import annotate, overlay_masks
from robokit.perception import GroundingDINOObjectPredictor, SAM2Predictor

# Define absl flags for CLI arguments
FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", None, "Directory path to input video frames")



def create_symlink_for_gt_masks(video_dir):
    target_dir = os.path.abspath(os.path.join(video_dir, "../gsam2/masks"))
    link_name = os.path.join(video_dir, "..", "gt_masks")

    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target mask directory does not exist: {target_dir}")

    # Remove existing symlink if it exists
    if os.path.islink(link_name) or os.path.exists(link_name):
        os.remove(link_name)
        print(f"🔄 Removed existing link or directory: {link_name}")

    os.symlink(target_dir, link_name)
    print(f"✅ Created symlink: {link_name} → {target_dir}")


def load_palette_from_url(
    url="https://raw.githubusercontent.com/IRVLUTD/fewsol-toolkit/refs/heads/main/palette.txt",
):
    """
    Load the color palette from a remote URL.

    Args:
        url (str): URL to the palette.txt file.

    Returns:
        list: List of (R, G, B) tuples.
    """
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    palette = [
        tuple(map(int, line.strip().split()))
        for line in response.text.strip().split("\n")
    ]
    return palette


def apply_palette(mask, palette):
    """
    Apply a color palette to a segmentation mask.

    Args:
        mask (torch.Tensor): Tensor of shape [H, W] containing integer labels.
        palette (list): List of (R, G, B) tuples.

    Returns:
        np.ndarray: RGB image of shape [H, W, 3].
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = torch.unique(mask)
    for label in unique_labels:
        if label == 0:
            continue  # Skip background (assuming background is 0)
        # Ensure label is an integer before indexing the palette
        label_int = int(label.item())  # Convert to int explicitly
        color = palette[label_int % len(palette)]  # Cycle through palette if needed
        colored_mask[mask == label_int] = color

    return PILImg.fromarray(colored_mask)


def combine_masks(gt_masks):
    """
    Combine several bit masks [N, H, W] into a mask [H,W],
    e.g. 8*480*640 tensor becomes a numpy array of 480*640.
    [[1,0,0], [0,1,0]] = > [1,2,0].

    Args:
        gt_masks (torch.Tensor): Tensor of shape [N, H, W] representing multiple bit masks.

    Returns:
        torch.Tensor: Combined mask of shape [H, W].
    """
    try:
        gt_masks = torch.flip(gt_masks, dims=(0,))
        num, h, w = gt_masks.shape
        bin_mask = torch.zeros((h, w), device=gt_masks.device)
        num_instance = len(gt_masks)

        # if there is not any instance, just return a mask full of 0s.
        if num_instance == 0:
            return bin_mask

        for m, object_label in zip(gt_masks, range(1, 1 + num_instance)):
            label_pos = torch.nonzero(m, as_tuple=True)
            bin_mask[label_pos] = object_label
        return bin_mask

    except Exception as e:
        logging.error(f"Error combining masks: {e}")
        raise e


def remove_n_largest_bboxes(initial_bboxes, phrases, gdino_conf, n):
    """
    Remove the 'n' bounding boxes with the largest areas and their corresponding phrases and confidence scores.

    Args:
        initial_bboxes (torch.Tensor): Tensor of shape [N, 4] representing bounding boxes (x, y, w, h).
        phrases (list): List of phrases corresponding to each bounding box.
        gdino_conf (torch.Tensor): Tensor of confidence scores for each bounding box.
        n (int): Number of largest bounding boxes to remove.

    Returns:
        Tuple: Updated initial_bboxes, phrases, and gdino_conf with the 'n' largest bboxes removed.
    """
    if n <= 0:
        return initial_bboxes, phrases, gdino_conf

    # Compute the area of each bounding box (w * h)
    bbox_areas = initial_bboxes[:, 2] * initial_bboxes[:, 3]  # width * height

    # Get indices of the 'n' largest bounding boxes
    largest_indices = torch.argsort(bbox_areas, descending=True)[:n]

    # Create a mask to keep only the non-largest bounding boxes
    keep_indices = torch.ones(len(initial_bboxes), dtype=torch.bool)
    keep_indices[largest_indices] = False  # Set the largest ones to False

    # Apply the mask to filter out the largest bounding boxes
    initial_bboxes = initial_bboxes[keep_indices]
    gdino_conf = gdino_conf[keep_indices]

    # Remove corresponding phrases
    phrases = [phrases[i] for i in range(len(phrases)) if keep_indices[i]]

    return initial_bboxes, phrases, gdino_conf


def merge_rgb_with_mask(rgb_pil, masks_pil):
    """
    Merges the RGB image with the mask while applying a semi-transparent black overlay to the background.

    Args:
        rgb_pil (PIL.Image): The RGB image.
        masks_pil (PIL.Image): The mask image with a palette.

    Returns:
        PIL.Image: Merged image with highlighted masks and darkened background.
    """
    # Convert images to RGBA
    rgb_image = rgb_pil.convert("RGB")
    mask_image = masks_pil.convert("RGBA")

    # Convert images to NumPy arrays
    rgb_array = np.array(rgb_image, dtype=np.float32)
    mask_array = np.array(mask_image, dtype=np.float32)

    # Detect non-black regions in the mask
    mask_binary = (mask_array[..., :3] != [0, 0, 0]).any(axis=-1)

    # Darken the background (apply transparency)
    bg_alpha = 0.2  # Transparency level for background
    rgb_darkened = rgb_array * (1 - bg_alpha)

    # Blend the mask with transparency
    mask_alpha = 0.65  # Transparency for the mask colors
    blended_mask = rgb_darkened.copy()
    blended_mask[mask_binary] = mask_array[mask_binary][
        :, :3
    ] * mask_alpha + rgb_darkened[mask_binary] * (1 - mask_alpha)

    # Convert back to PIL Image
    blended_image = PILImg.fromarray(np.uint8(blended_mask))

    return blended_image


def merge_rgb_with_mask_with_contours(rgb_pil, masks_pil, bg_alpha=0.4, mask_alpha=0.65):
    """
    Overlay segmentation masks onto an RGB image with white contours and a soft dark background.

    Args:
        rgb_pil (PIL.Image): RGB input image.
        masks_pil (PIL.Image): Color-palette mask image.
        bg_alpha (float): Darkness level of the background.
        mask_alpha (float): Transparency of mask colors.

    Returns:
        PIL.Image: Final composite image.
    """
    rgb_array = np.array(rgb_pil.convert("RGB"), dtype=np.float32)
    mask_rgba = np.array(masks_pil.convert("RGBA"), dtype=np.float32)
    mask_rgb = mask_rgba[..., :3]
    mask_binary = (mask_rgb != [0, 0, 0]).any(axis=-1)

    # Darken background
    rgb_darkened = rgb_array * (1 - bg_alpha)

    # Blend mask
    blended = rgb_darkened.copy()
    blended[mask_binary] = (
        mask_rgb[mask_binary] * mask_alpha +
        rgb_darkened[mask_binary] * (1 - mask_alpha)
    )

    # Add white contours
    gray_mask = np.array(masks_pil.convert("L"))
    for val in np.unique(gray_mask):
        if val == 0:
            continue
        mask = (gray_mask == val).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blended = cv2.drawContours(np.uint8(blended), contours, -1, (255, 255, 255), 2)

    return PILImg.fromarray(np.uint8(np.clip(blended, 0, 255)))


def main(argv):
    # Get input values from flags
    video_dir = sanity_check(argv)

    logging.info("Initialize object detectors")

    # Initialize SAM2 for tracking across frames
    sam2 = SAM2Predictor()

    mask_out_dir = os.path.join(video_dir, "../gsam2/masks/")
    overlay_out_dir = os.path.join(video_dir, "../gsam2/bbox_overlay/")
    palette_out_dir = os.path.join(video_dir, "../gsam2/palette/")
    rgbm_out_dir = os.path.join(video_dir, "../gsam2/rgb_and_mask/")

    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(overlay_out_dir, exist_ok=True)
    os.makedirs(palette_out_dir, exist_ok=True)
    os.makedirs(rgbm_out_dir, exist_ok=True)

    # Load the palette from the URL
    palette = load_palette_from_url()

    img_name = sorted(os.listdir(video_dir))[0]
    img_path = os.path.join(video_dir, img_name)

    # Read the first frame to detect initial bounding boxes
    img_pil = PILImg.open(img_path).convert("RGB")

    bbox_prompts_file = os.path.join(video_dir, "..", "prompts.json")

    bbox_prompts = None
    # Read the JSON file
    with open(bbox_prompts_file, 'r') as f:
        bbox_prompts = json.load(f)
        bbox_prompts = np.array(bbox_prompts["bboxes_xyxy"])

    # bboxes = np.array([
    #     [337, 167, 372, 235],
    #     [374, 184, 423, 235],
    #     [291, 188, 322, 240],
    #     [254, 175, 292, 241],
    #     [213, 182, 237, 228]
    # ])

    bboxes = np.array([
        [265, 208, 317, 234],
        [288, 213, 344, 250],
        [231, 206, 266, 233],
        [260, 173, 288, 216],
        [259, 230, 291, 258]
    ])



    video_segments = sam2.propagate_bbox_prompt_masks_and_save(video_dir, bbox_prompts)

    filenames = list(reversed(list(video_segments.keys())))

    for idx, (filename, masks) in tqdm(enumerate(video_segments.items())):

        # Remove the extra dimension (from shape (N, 1, H, W) -> (N, H, W))
        masks_squeezed = torch.tensor(masks).squeeze(1)

        # Generate the combined mask after removing the largest object
        combined_mask = combine_masks(masks_squeezed)

        img_pil = PILImg.open(os.path.join(video_dir, filename)).convert("RGB")

        img_name = filenames[idx].replace("jpg", "png")

        if idx == 0:
            # Apply overlay and annotations only on first frame
            gdino_conf = [1.0 for i in bbox_prompts]
            phrases = ["object" for i in bbox_prompts]
            bbox_annotated_pil = annotate(
                overlay_masks(img_pil, masks_squeezed),
                bbox_prompts,
                gdino_conf,
                phrases,
            )  # Annotate
            bbox_annotated_pil.save(os.path.join(overlay_out_dir, img_name))

        # Convert to numpy
        combined_mask_np = np.array(
            combined_mask.cpu(), dtype=np.uint8
        )  # Ensure it's uint8

        # Convert to PIL Image in 'L' mode (grayscale)
        to_save = PILImg.fromarray(combined_mask_np, mode="L")
        to_save.save(os.path.join(mask_out_dir, img_name))

        # Apply the palette
        colored_mask = apply_palette(combined_mask, palette)
        colored_mask.save(os.path.join(palette_out_dir, img_name))

        # combined_image = merge_rgb_with_mask(img_pil, colored_mask)
        combined_image = merge_rgb_with_mask_with_contours(img_pil, colored_mask)

        combined_image.save(os.path.join(rgbm_out_dir, img_name))

    create_symlink_for_gt_masks(os.path.join(video_dir))
    print("🎉 All masks propagated and saved successfully!")


def sanity_check(argv):
    input_dir = flags.FLAGS.input_dir
    rgb_dir = os.path.join(input_dir, "jpg/")
    
    # Check if the input directory exists and is a valid directory
    if not os.path.isdir(rgb_dir):
        raise Exception(
            f"Error: The directory '{rgb_dir}' does not exist or is not a valid directory."
        )

    # List all files in the directory and filter for image files (jpg, jpeg, png)
    image_files = [
        f
        for f in os.listdir(rgb_dir)
        if os.path.isfile(os.path.join(rgb_dir, f))
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        raise Exception(f"No image files found in the directory '{rgb_dir}'.")

    return rgb_dir


if __name__ == "__main__":
    # Mark flags as required
    flags.mark_flag_as_required("input_dir")

    # Run the main function
    app.run(main)
