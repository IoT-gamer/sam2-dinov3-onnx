# -*- coding: utf-8 -*-
"""
Hybrid EdgeTAM + DINOv3 Object Tracker

This script tracks an object in a video using a combination of two models:
1. DINOv3: Acts as a "scout" to find the approximate location of the object in each frame.
2. EdgeTAM: Acts as an "artist" to paint a precise, high-quality segmentation mask
   based on a centroid point derived from DINOv3's rough mask.

The final output is a video file with the high-quality mask overlayed on each frame.
"""

import os
import cv2
import math
import functools
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from typing import Tuple, List

# --- Constants and Hyperparameters ---
# Model paths
EDGETAM_ENCODER_PATH = "../models/edgetam_encoder.onnx"
EDGETAM_DECODER_PATH = "../models/edgetam_decoder.onnx"
DINO_MODEL_PATH = "dinov3_feature_extractor.onnx"

# Video I/O
INPUT_VIDEO_PATH = "05_default_juggle.mp4" # https://github.com/facebookresearch/EdgeTAM/blob/main/examples/05_default_juggle.mp4
OUTPUT_VIDEO_PATH = "output_tracking.mp4"

# EdgeTAM settings
EDGETAM_INPUT_SIZE = 1024

# DINOv3 settings
DINO_PATCH_SIZE = 16
DINO_SHORT_SIDE_RES = 180  # Using a smaller resolution for speed

# Tracking algorithm settings
MAX_CONTEXT_LENGTH = 7
NEIGHBORHOOD_SIZE = 12
NEIGHBORHOOD_SHAPE = "circle"
TOPK = 4
TEMPERATURE = 0.1

# --- Helper Functions for EdgeTAM ---

def preprocess_image_edgetam(image_array: np.ndarray, input_size: int = 1024) -> np.ndarray:
    """Resizes, pads, and normalizes an image for EdgeTAM ONNX model inference."""
    orig_height, orig_width, _ = image_array.shape
    resized_width, resized_height = input_size, input_size

    input_array_resized = cv2.resize(image_array, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    # Normalize with ImageNet stats
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    input_tensor = (input_array_resized - mean) / std

    # Transpose to CHW format and add batch dimension
    input_tensor = input_tensor.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    return input_tensor

def preprocess_point_edgetam(
    point: np.ndarray,
    label: np.ndarray,
    orig_size: Tuple[int, int],
    resized_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesses a point for EdgeTAM ONNX model inference."""
    orig_height, orig_width = orig_size
    resized_height, resized_width = resized_size

    onnx_coord = np.concatenate([point, np.array([[0.0, 0.0]])], axis=0)[None, :, :].astype(np.float32)
    onnx_label = np.concatenate([label, np.array([-1])])[None, :].astype(np.float32)

    # Scale coordinates to the resized image dimensions
    onnx_coord[..., 0] = onnx_coord[..., 0] * (resized_width / orig_width)
    onnx_coord[..., 1] = onnx_coord[..., 1] * (resized_height / orig_height)
    return onnx_coord, onnx_label

def run_inference_edgetam(
    encoder_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    image_tensor: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    original_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs EdgeTAM inference and handles mask upscaling."""
    # 1. Encoder inference
    encoder_outputs = encoder_session.run(None, {'image': image_tensor})
    image_embed, high_res_feats_0, high_res_feats_1, _ = encoder_outputs

    # 2. Decoder inference
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    decoder_outputs = decoder_session.run(None, {
        'image_embed': image_embed, 'high_res_feats_0': high_res_feats_0,
        'high_res_feats_1': high_res_feats_1, "point_coords": point_coords,
        "point_labels": point_labels, "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
    })
    low_res_masks, iou_predictions = decoder_outputs

    # 3. Post-processing: Select the best mask and resize it
    best_mask_idx = np.argmax(iou_predictions[0])
    selected_low_res_mask = low_res_masks[0, best_mask_idx, :, :]
    
    # Use OpenCV to resize the mask to the original image's dimensions
    resized_mask = cv2.resize(
        selected_low_res_mask,
        (original_size[1], original_size[0]), # cv2 expects (width, height)
        interpolation=cv2.INTER_LINEAR
    )
    return resized_mask, iou_predictions

# --- Helper Functions for DINOv3 ---

def preprocess_frame_dino(frame: np.ndarray, short_side: int, patch_size: int) -> np.ndarray:
    """Resizes, normalizes, and prepares a video frame for DINOv3 ONNX model."""
    old_height, old_width, _ = frame.shape

    def _round_up(side: float) -> int:
        return math.ceil(side / patch_size) * patch_size

    if old_width > old_height:
        new_height = _round_up(short_side)
        new_width = _round_up(old_width * new_height / old_height)
    else:
        new_width = _round_up(short_side)
        new_height = _round_up(old_height * new_width / old_width)

    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img_fp32 = resized_frame.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized_frame = (img_fp32 - mean) / std

    return normalized_frame.transpose(2, 0, 1)

def forward_onnx_dino(session: ort.InferenceSession, img_numpy: np.ndarray) -> np.ndarray:
    """Runs inference using the DINO ONNX session and post-processes the features."""
    input_name = session.get_inputs()[0].name
    img_batch = np.expand_dims(img_numpy, axis=0) if len(img_numpy.shape) == 3 else img_numpy

    # Calculate patch dimensions from the input image shape
    _, H_img, W_img = img_numpy.shape
    H_patches = H_img // DINO_PATCH_SIZE
    W_patches = W_img // DINO_PATCH_SIZE

    # Run inference. The result shape is (1, num_patches, channels)
    results = session.run(None, {input_name: img_batch})[0]

    # Squeeze the batch dimension and reshape to a (H, W, C) grid
    features_flat = np.squeeze(results, axis=0)
    result_tensor = features_flat.reshape(H_patches, W_patches, -1)
    
    # Normalize features (L2 norm)
    norm = np.linalg.norm(result_tensor, axis=-1, keepdims=True)
    return result_tensor / (norm + 1e-6)

@functools.lru_cache()
def make_neighborhood_mask_numpy(h: int, w: int, size: float, shape: str) -> np.ndarray:
    """Creates a neighborhood mask using NumPy."""
    coords_y, coords_x = np.mgrid[0:h, 0:w].astype(np.float32)
    ij = np.stack([coords_y, coords_x], axis=-1)

    ord = 2 if shape == "circle" else np.inf
    diff = ij[:, :, np.newaxis, np.newaxis, :] - ij[np.newaxis, np.newaxis, :, :, :]
    norm = np.linalg.norm(diff, ord=ord, axis=-1)
    return norm <= size

def propagate_numpy(
    current_features: np.ndarray, context_features: np.ndarray,
    context_probs: np.ndarray, neighborhood_mask: np.ndarray,
    topk: int, temperature: float
) -> np.ndarray:
    """Propagates masks using NumPy operations."""
    t, h, w, M = context_probs.shape
    h_curr, w_curr, _ = current_features.shape

    dot = np.einsum("ijd,tuvd->ijtuv", current_features, context_features)
    dot = np.where(neighborhood_mask[:, :, np.newaxis, :, :], dot, -np.inf)
    dot_flat = dot.reshape(h_curr * w_curr, -1)

    k_th_largest = np.partition(dot_flat, -topk, axis=1)[:, -topk]
    dot_flat = np.where(dot_flat >= k_th_largest[:, np.newaxis], dot_flat, -np.inf)

    weights = softmax(dot_flat / temperature, axis=1)
    context_probs_flat = context_probs.reshape(-1, M)
    current_probs_flat = weights @ context_probs_flat
    current_probs_flat /= current_probs_flat.sum(axis=1, keepdims=True)

    return current_probs_flat.reshape(h_curr, w_curr, M)

# --- Visualization Utility ---

def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Converts a binary segmentation mask to a colorized RGB image."""
    # This creates a red overlay for the mask
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb_mask[mask == 1] = [255, 0, 0] # Red for the segmented object
    return rgb_mask

# --- Main Execution ---

def main():
    """Main function to run the tracking process."""
    # 1. Load ONNX Models
    print("Loading ONNX models...")
    edgetam_encoder_session = ort.InferenceSession(EDGETAM_ENCODER_PATH)
    edgetam_decoder_session = ort.InferenceSession(EDGETAM_DECODER_PATH)
    dino_session = ort.InferenceSession(DINO_MODEL_PATH)
    print("Models loaded successfully.")

    # 2. Setup Video I/O
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: Input video not found at '{INPUT_VIDEO_PATH}'")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{INPUT_VIDEO_PATH}'")
        return

    # Get video properties for the output writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"Output will be saved to '{OUTPUT_VIDEO_PATH}'")


    # 3. Initialization on the First Frame
    print("Processing first frame for initialization...")
    ret, first_frame_bgr = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return
        
    first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    orig_size = (frame_height, frame_width)

    # EdgeTAM: Get the high-quality first mask with an initial point prompt
    initial_point = np.array([[645, 315]])  # Example: user's first click
    initial_label = np.array([1])
    input_tensor_edgetam = preprocess_image_edgetam(first_frame_rgb, EDGETAM_INPUT_SIZE)
    onnx_coord, onnx_label = preprocess_point_edgetam(
        initial_point, initial_label, orig_size, (EDGETAM_INPUT_SIZE, EDGETAM_INPUT_SIZE)
    )
    first_mask_np, _ = run_inference_edgetam(
        edgetam_encoder_session, edgetam_decoder_session, 
        input_tensor_edgetam, onnx_coord, onnx_label, orig_size
    )
    first_mask_np = (first_mask_np > 0).astype(np.uint8)  # Binarize the mask

    # DINOv3: Get the first set of features
    dino_input = preprocess_frame_dino(first_frame_rgb, DINO_SHORT_SIDE_RES, DINO_PATCH_SIZE)
    first_feats = forward_onnx_dino(dino_session, dino_input)
    _, H_dino, W_dino = dino_input.shape
    feats_height, feats_width = H_dino // DINO_PATCH_SIZE, W_dino // DINO_PATCH_SIZE

    # Downscale the EdgeTAM mask to DINO feature size for context
    first_mask_resized_dino = cv2.resize(
        first_mask_np, (feats_width, feats_height), interpolation=cv2.INTER_NEAREST
    )
    num_masks = int(first_mask_np.max()) + 1
    first_probs = np.eye(num_masks)[first_mask_resized_dino]

    # 4. Setup for Tracking Loop
    features_queue: List[np.ndarray] = []
    probs_queue: List[np.ndarray] = []
    neighborhood_mask = make_neighborhood_mask_numpy(feats_height, feats_width, NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SHAPE)

    # 5. Main Tracking Loop
    frame_idx = 0
    while True:
        ret, current_frame_bgr = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        print(f"Processing frame {frame_idx}...")

        current_frame_rgb = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB)

        # Step A: DINOv3 Scout finds the rough location
        dino_input = preprocess_frame_dino(current_frame_rgb, DINO_SHORT_SIDE_RES, DINO_PATCH_SIZE)
        current_feats = forward_onnx_dino(dino_session, dino_input)

        context_feats = np.stack([first_feats, *features_queue], axis=0)
        context_probs = np.stack([first_probs, *probs_queue], axis=0)
        
        current_probs = propagate_numpy(current_feats, context_feats, context_probs, neighborhood_mask, TOPK, TEMPERATURE)

        # Upsample DINO mask to find the centroid
        dino_probs_mhw = current_probs.transpose(2, 0, 1)
        upsampled_probs = np.array([cv2.resize(p, orig_size[::-1], interpolation=cv2.INTER_LINEAR) for p in dino_probs_mhw])
        dino_pred_mask = np.argmax(upsampled_probs, axis=0).astype(np.uint8)

        # Step B: Calculate Centroid from DINO's rough mask
        M = cv2.moments(dino_pred_mask)
        overlayed_frame = current_frame_bgr # Default to original frame if no contour found
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Step C: EdgeTAM Artist paints the detailed mask using the centroid
            edgetam_input_tensor = preprocess_image_edgetam(current_frame_rgb, EDGETAM_INPUT_SIZE)
            new_point = np.array([[cX, cY]])
            new_label = np.array([1])
            onnx_coord_new, onnx_label_new = preprocess_point_edgetam(
                new_point, new_label, orig_size, (EDGETAM_INPUT_SIZE, EDGETAM_INPUT_SIZE)
            )
            final_mask, _ = run_inference_edgetam(
                edgetam_encoder_session, edgetam_decoder_session,
                edgetam_input_tensor, onnx_coord_new, onnx_label_new, orig_size
            )
            final_binary_mask = (final_mask > 0).astype('uint8')

            # --- Visualization ---
            final_mask_rgb = mask_to_rgb(final_binary_mask)
            overlayed_frame = cv2.addWeighted(current_frame_bgr, 0.7, cv2.cvtColor(final_mask_rgb, cv2.COLOR_RGB2BGR), 0.3, 0)
            cv2.circle(current_frame_bgr, (cX, cY), 10, (0, 255, 0), -1) # Green dot for centroid

        # Write the frame to the output video
        video_writer.write(overlayed_frame)

        # Step D: Update DINO context for the next frame
        features_queue.append(current_feats)
        probs_queue.append(current_probs)
        if len(features_queue) > MAX_CONTEXT_LENGTH:
            features_queue.pop(0)
        if len(probs_queue) > MAX_CONTEXT_LENGTH:
            probs_queue.pop(0)

    # 6. Release Resources
    print("Processing complete. Releasing resources.")
    cap.release()
    video_writer.release()
    print(f"Video saved successfully to '{OUTPUT_VIDEO_PATH}'")

if __name__ == "__main__":
    main()
