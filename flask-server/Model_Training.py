#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import argparse
from multiprocessing import freeze_support
import torch
import cv2
import pickle
import random
import os
import numpy as np
from torch import nn
import torch
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEModel, VideoMAEConfig, Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig
from multisports_visualizer import (
    plot_class_distribution,
    plot_framewise_labels,
    preview_clip_labels,
    inspect_pool_distribution
)
import torch.nn as nn
from transformers import VideoMAEConfig, VideoMAEModel


print("GPU Available:", torch.cuda.is_available())
print("Current Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


def labelcheck(frame_labels):
  for labels in frame_labels:
    total = 0
    for frame in range(len(frame_labels[labels])):
      total = total + frame_labels[labels][frame].sum()
    print(f"{labels} Has a total of {len(frame_labels[labels])} frames and {float(total)} frames with classes")


def generate_balanced_clip_index_pool(frame_labels, clip_len=16, stride=4):
    pos_pool = []
    neg_pool = []

    for vid, label_matrix in frame_labels.items():
        num_frames = label_matrix.shape[0]
        for start in range(0, num_frames - clip_len + 1, stride):
            clip_labels = label_matrix[start : start + clip_len]
            if clip_labels.sum() > 0:
                pos_pool.append((vid, start))
            else:
                neg_pool.append((vid, start))

    # Balance the pools
    min_len = min(len(pos_pool), len(neg_pool))
    balanced_pool = pos_pool[:min_len] + neg_pool[:min_len]
    np.random.shuffle(balanced_pool)
    return balanced_pool

# Image normalization stats (ImageNet means and stds)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

class BalancedMultiSportsDataset(Dataset):
    def __init__(self, frame_labels, video_dir, clip_len=16, is_train=True,
                 min_label_count=1, balance_strategy="upsample", background_class_idx=-1):
        """
        Args:
            frame_labels (dict): video_id -> (num_frames x num_classes+1) labels
            video_dir (str): path to video files
            clip_len (int): frames per sample
            is_train (bool): toggle augmentations or randomness
            min_label_count (int): minimum number of labeled frames (excluding background)
            balance_strategy (str): 'upsample' or 'uniform'
            background_class_idx (int): index of the background column in label matrix
        """
        self.frame_labels = frame_labels
        self.video_dir = video_dir
        self.clip_len = clip_len
        self.is_train = is_train
        self.balance_strategy = balance_strategy
        self.min_label_count = min_label_count

        # Total classes includes background
        self.total_classes = list(frame_labels.values())[0].shape[1]
        self.bg_class = background_class_idx if background_class_idx >= 0 else self.total_classes - 1
        self.num_classes = self.total_classes - 1  # exclude background from balancing

        # Build list of balanced (video_id, start_frame) samples
        self.sample_pool = self.build_balanced_sample_pool()

    def build_balanced_sample_pool(self):
        class_to_samples = {i: [] for i in range(self.num_classes)}  # ignore background class

        for vid, labels in self.frame_labels.items():
            nF = labels.shape[0]
            for start in range(0, nF - self.clip_len + 1, self.clip_len // 2):
                clip_labels = labels[start:start + self.clip_len]
                action_sum = clip_labels[:, :self.num_classes].sum(axis=0)

                if action_sum.sum() < self.min_label_count:
                    continue  # skip background-only clips

                for cls in np.where(action_sum > 0)[0]:
                    class_to_samples[cls].append((vid, start))

        # Balance clips
        all_samples = []
        max_count = max(len(samples) for samples in class_to_samples.values() if samples)

        for cls, samples in class_to_samples.items():
            if not samples:
                continue
            if self.balance_strategy == "upsample":
                reps = max_count // len(samples) + 1
                all_samples.extend((samples * reps)[:max_count])
            elif self.balance_strategy == "uniform":
                all_samples.extend(random.sample(samples, min(len(samples), max_count)))
            else:
                all_samples.extend(samples)

        random.shuffle(all_samples)
        return all_samples

    def __len__(self):
        return len(self.sample_pool)

    def __getitem__(self, idx):
        vid, start = self.sample_pool[idx]
        labels = self.frame_labels[vid]
        nF = labels.shape[0]

        video_path = f"{self.video_dir}/{vid}.mp4"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        read_count = 0
        while read_count < self.clip_len:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame
            frame = cv2.resize(frame, (224, 224))
            frame = frame[:, :, ::-1].astype(np.float32) / 255.0
            frame = (frame - IMG_MEAN) / IMG_STD
            frames.append(frame)
            read_count += 1
        cap.release()

        # Pad if needed
        if len(frames) < self.clip_len:
            last = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.float32)
            frames += [last.copy()] * (self.clip_len - len(frames))

        frames = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)

        # Get label slice
        clip_labels = labels[start:start + self.clip_len]
        if clip_labels.shape[0] < self.clip_len:
            pad_rows = self.clip_len - clip_labels.shape[0]
            clip_labels = np.concatenate(
                [clip_labels, np.zeros((pad_rows, self.total_classes), dtype=np.float32)],
                axis=0
            )

        labels_tensor = torch.from_numpy(clip_labels)
        mask = torch.ones(self.clip_len, dtype=torch.float32)
        if clip_labels.shape[0] < self.clip_len:
            mask[clip_labels.shape[0]:] = 0.0

        return frames_tensor, labels_tensor, mask, vid

class VideoMAEActionDetector(nn.Module):
    def __init__(self, num_classes, pretrained_model_name="MCG-NJU/videomae-large"):
        super().__init__()
        # Load VideoMAE with tubelet_size=1 and pretrained weights
        config = VideoMAEConfig.from_pretrained(pretrained_model_name)
        config.num_frames = 16           # we will input 16 frames
        config.tubelet_size = 1          # adjust tubelet to 1 for per-frame tokens
        config.use_mean_pooling = False  # we'll do our own pooling (not just a single mean for whole video)
        self.backbone = VideoMAEModel.from_pretrained(pretrained_model_name, config=config, ignore_mismatched_sizes=True)
        hidden_dim = config.hidden_size  # typically 768 for ViT-Base
        self.cls_head = nn.Linear(hidden_dim, num_classes)  # classifier for each frame

    def forward(self, video_frames):
        """
        video_frames: Tensor of shape (B, T, C, H, W), where T=16, C=3, H=W=224
        """
        B, T, C, H, W = video_frames.shape
        # Forward through VideoMAE backbone
        outputs = self.backbone(pixel_values=video_frames, bool_masked_pos=None, return_dict=True)
        # outputs.last_hidden_state: shape (B, seq_len, hidden)
        # For tubelet=1, seq_len = T * (patch_count). patch_count = 196 for 224x224 with 16x16 patches.
        seq_output = outputs.last_hidden_state  # (B, T*patch_count, hidden)
        # Reshape to separate frames
        # Note: If tubelet_size were >1, T would effectively be T/tubelet in this shape.
        frame_features = seq_output.view(B, T, -1, seq_output.size(-1))  # (B, T, patch_count, hidden)
        # Average pool over spatial patches to get a feature per frame
        frame_features = frame_features.mean(dim=2)  # (B, T, hidden)
        # Apply classifier head to each frame feature
        logits = self.cls_head(frame_features)  # (B, T, num_classes)
        return logits

def compute_class_ap(class_idx, predictions, ground_truths, iou_thresh):
    """Compute AP for a single class at the given IoU threshold."""
    preds = predictions[class_idx]
    gts = ground_truths[class_idx]
    if not gts:
        return None  # no ground truth for this class (could skip or return None)
    # Count GT segments
    n_gt = len(gts)
    # Sort predictions by confidence
    preds = sorted(preds, key=lambda x: x[0], reverse=True)
    # Track matched GTs to avoid double matching
    matched = set()
    tp = np.zeros(len(preds), dtype=np.float32)
    fp = np.zeros(len(preds), dtype=np.float32)
    # For each prediction, decide TP or FP
    for i, (score, vid, p_start, p_end) in enumerate(preds):
        # Find best matching GT for this prediction (by IoU)
        best_iou = 0.0
        best_j = -1
        for j, (gt_vid, g_start, g_end) in enumerate(gts):
            if gt_vid != vid or j in matched:
                continue  # different video or already matched
            # Compute IoU
            inter_start = max(p_start, g_start)
            inter_end   = min(p_end, g_end)
            if inter_end < inter_start:
                iou = 0.0
            else:
                inter_len = inter_end - inter_start + 1
                union_len = (p_end - p_start + 1) + (g_end - g_start + 1) - inter_len
                iou = inter_len / union_len
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            # True Positive
            tp[i] = 1
            matched.add(best_j)
        else:
            # False Positive
            fp[i] = 1
    # Compute precision-recall
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / n_gt
    precisions = cum_tp / (cum_tp + cum_fp + 1e-10)
    # Compute AP as area under PR curve (trapz or vocational 11-pt interpolation)
    # We'll do a trapz integration over recall.
    # First, ensure arrays start with (0,1) and end with (1,0) for integration completeness
    recall_points = np.concatenate(([0.0], recalls, [recalls[-1]]))
    precision_points = np.concatenate(([1.0], precisions, [0.0]))
    # Make precision non-decreasing when going backward (monotonicity fix for AP calc)
    for k in range(len(precision_points)-2, -1, -1):
        precision_points[k] = max(precision_points[k], precision_points[k+1])
    # Compute AP as sum of (recall change * precision) across recall points
    ap = 0.0
    for k in range(1, len(recall_points)):
        delta_r = recall_points[k] - recall_points[k-1]
        ap += precision_points[k] * delta_r
    return ap

def make_pos_weight(frame_labels, bg_class_idx):
    num_classes = list(frame_labels.values())[0].shape[1]
    action_classes = list(range(num_classes))
    action_classes.remove(bg_class_idx)

    total_frames = 0
    class_counts = np.zeros(num_classes)
    for labels in frame_labels.values():
        total_frames += labels.shape[0]
        class_counts += labels.sum(axis=0)

    class_freq = class_counts / total_frames
    pos_weights = 1.0 / np.clip(class_freq, 1e-6, None)

    # Optional: downweight background
    pos_weights[bg_class_idx] = 0.1 * pos_weights[bg_class_idx]

    return torch.tensor(pos_weights, dtype=torch.float32)


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Train VideoMAE on temporal action detection")
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing .mp4 video files')
    parser.add_argument('--pickle_file', type=str, required=True, help='Path to .pkl file with ground truth labels')
    parser.add_argument('--max_steps', type=int, required=False,help='Max steps per epoch')
    args = parser.parse_args()


    video_dir = args.video_dir
    pickle_file = video_dir + args.pickle_file
    max_steps = args.max_steps

    ann_file = pickle_file

    with open(ann_file, 'rb') as f:
        data = pickle.load(f)

    # Assuming data is the loaded pickle dict
    labels = data['labels']              # list of class names
    train_ids = data['train_videos'][0]  # list of training video IDs
    test_ids  = data['test_videos'][0]   # list of test video IDs
    gttubes = data['gttubes']            # ground-truth tubes dict
    nframes_dict = data['nframes']

    # Create a dict: video_id -> [num_frames x num_classes] binary array
    frame_labels = {}
    num_classes = len(labels)
    for vid in (*train_ids, *test_ids):
        nF = nframes_dict[vid]
        frame_label_array = np.zeros((nF, num_classes + 1), dtype=np.float32)
        # fill in labels for each tube
        for class_idx, tubes in gttubes[vid].items():
            for tube in tubes:  # tube is an array of shape (m,5)
                # tube[:,0] are frame indices where this action instance appears (1-indexed in GT)
                frames = tube[:,0].astype(int)  # frame numbers
                # Convert to 0-indexed frame indices in array
                for f in frames:
                    if 1 <= f <= nF:
                        frame_label_array[f-1, class_idx] = 1.0
        frame_labels[vid] = frame_label_array

    action_labels = frame_label_array[:, :num_classes]
    background = (action_labels.sum(axis=1) == 0).astype(np.float32)
    frame_label_array[:, -1] = background
    class_names = data['labels']
    labels.append("background")

    train_dataset = BalancedMultiSportsDataset(
    frame_labels=frame_labels,
    video_dir=video_dir,
    clip_len=16,
    is_train=True,
    min_label_count=1,  # only use clips with at least 1 labeled action frame
    balance_strategy="upsample",  # or "uniform"
    background_class_idx=len(labels) - 1  # assuming 'background' is last
)

    val_dataset = BalancedMultiSportsDataset(
    frame_labels=frame_labels,
    video_dir=video_dir,
    clip_len=16,
    is_train=False,
    min_label_count=1,
    balance_strategy="none",  # keep original sampling
    background_class_idx=len(labels) - 1
    )

    
    plot_class_distribution(frame_labels, class_names)
    #plot_framewise_labels("basketball/v_00HRwkvvjtQ_c001", frame_labels, class_names)
    preview_clip_labels(train_dataset, class_names, num_samples=5)
    inspect_pool_distribution(train_dataset.sample_pool, frame_labels, clip_len=16)

    num_classes = len(labels)  # e.g., 66 for full MultiSports
    model = VideoMAEActionDetector(num_classes).cuda()


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    val_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    bg_class_idx = len(class_names) - 1  # background is last
    pos_weight = make_pos_weight(frame_labels, bg_class_idx)

    # Now use it in the loss
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none').cuda()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Learning rate scheduler: cosine with warmup
    num_epochs = 1
    train_steps_per_epoch = len(train_loader)  # one clip per video per epoch in our setting
    total_train_steps = num_epochs * train_steps_per_epoch
    warmup_steps = int(0.1 * total_train_steps)  # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

    # train on GPU
    scaler = GradScaler()
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (frames, labels, mask, vid) in enumerate(train_loader):
            frames = frames.cuda(non_blocking=True).float()       # shape (B, T, C, H, W)
            labels = labels.cuda(non_blocking=True).float()       # shape (B, T, num_classes)
            mask   = mask.cuda(non_blocking=True).float()         # shape (B, T)

            optimizer.zero_grad()
            with autocast():
                logits = model(frames)                   # (B, T, num_classes)
                # Compute elementwise loss
                elem_loss = loss_fn(logits, labels)      # (B, T, num_classes)
                # Apply mask: expand mask to shape (B, T, num_classes) for broadcasting
                mask_exp = mask.unsqueeze(-1)            # (B, T, 1)
                masked_loss = elem_loss * mask_exp       # zero-out loss where mask is 0
                # Compute average loss over *valid frames*
                # Use mask.sum() * num_classes as divisor if we want average per label per frame,
                # but better to average over frames then over classes (both ways equivalent if weighted properly).
                if mask.sum() > 0:
                    loss = masked_loss.sum() / (mask.sum() * masked_loss.size(-1))
                else:
                    # if no real frames in batch (unlikely in training), just average normally
                    loss = masked_loss.mean()

            # Backpropagate with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            # Print training info every N batches
            if (i + 1) % 10 == 0:
                avg_loss = running_loss / 10
                print(f"Epoch {epoch+1} - Batch {i+1}/{len(train_loader)} - Avg Loss: {avg_loss:.4f}")
                running_loss = 0.0

            if i + 1 >= max_steps:
                print(f"Reached step limit of {max_steps} — breaking early.")
                break


    save_path = "C:/Users/ridas/WebstormProjects/ReactFlask/flask-server/videomae_temporal_detector_Big1.pth"
    torch.save(model.state_dict(), save_path)