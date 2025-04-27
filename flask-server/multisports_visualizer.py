
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_class_distribution(frame_labels, class_names):
    class_totals = np.zeros(len(class_names))
    for labels in frame_labels.values():
        class_totals += labels.sum(axis=0)

    plt.figure(figsize=(14,5))
    plt.bar(range(len(class_names)), class_totals)
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.title("Total # of frames per class across all videos")
    plt.ylabel("Frame Count")
    plt.tight_layout()
    plt.show()

def plot_framewise_labels(vid, frame_labels, class_names):
    label_matrix = frame_labels[vid]
    plt.figure(figsize=(14, 4))
    plt.imshow(label_matrix.T, aspect='auto', interpolation='nearest', cmap='hot')
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Frame")
    plt.title(f"Frame-wise class activation map for {vid}")
    plt.colorbar(label="Label presence")
    plt.tight_layout()
    plt.show()

def preview_clip_labels(dataset, class_names, num_samples=5):
    for i in range(num_samples):
        frames, labels, mask, vid = dataset[i]
        label_counts = labels.sum(dim=0).numpy()
        active_classes = [class_names[i] for i, v in enumerate(label_counts) if v > 0]
        print(f"Sample {i}: Video = {vid}, Active Classes: {active_classes}")
        frame_img = frames[0].permute(1, 2, 0).numpy()
        frame_img = (frame_img * 0.229 + 0.485).clip(0, 1)  # Undo normalization for display
        plt.imshow(frame_img)
        plt.title(f"Clip starting at frame {i * dataset.clip_len}")
        plt.axis('off')
        plt.show()

def inspect_pool_distribution(pool, frame_labels, clip_len):
    pos = 0
    for vid, start in pool:
        if frame_labels[vid][start:start+clip_len].sum() > 0:
            pos += 1
    total = len(pool)
    print(f"{pos}/{total} clips contain action — {100*pos/total:.2f}% positive")
