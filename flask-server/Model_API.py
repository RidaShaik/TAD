import cv2
import numpy as np
from transformers import VideoMAEModel, VideoMAEConfig, Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig
from torch import nn
import torch
import pickle
from sklearn.metrics import f1_score

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

THRESHOLDS = np.float64(0.45), np.float64(0.73), np.float64(0.01), np.float64(0.79), np.float64(0.7100000000000001), np.float64(0.8300000000000001), np.float64(0.9500000000000001), np.float64(0.93), np.float64(0.65), np.float64(0.97), np.float64(0.97), np.float64(0.73), np.float64(0.75), np.float64(0.75), np.float64(0.01), np.float64(0.85), np.float64(0.99), np.float64(0.99), np.float64(0.13)

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

def infer_video_fast(model, video_path, num_classes=66, window_size=16, stride=8):
    model.eval()
    device = next(model.parameters()).device
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = frame[:, :, ::-1].astype(np.float32) / 255.0
        frame = (frame - IMG_MEAN) / IMG_STD
        frames.append(frame)
    cap.release()

    if not frames:
        return np.empty((0, num_classes), dtype=np.float32)

    frames = np.stack(frames, axis=0)
    nF = len(frames)
    scores_accum = np.zeros((nF, num_classes), dtype=np.float32)
    counts = np.zeros(nF, dtype=np.int32)

    for start in range(0, nF, stride):
        end = min(start + window_size, nF)
        clip = frames[start:end]
        if len(clip) < window_size:
            clip = np.concatenate([clip, np.repeat(clip[-1:], window_size - len(clip), axis=0)])
        x = torch.from_numpy(clip).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        for i, f in enumerate(range(start, end)):
            scores_accum[f] += probs[i]
            counts[f] += 1

    counts[counts == 0] = 1
    return scores_accum / counts[:, None]

def overlay_labels_on_video(video_path, pred_binary, pred_scores, class_names, 
                             output_path="annotated_output.mp4", font_scale=0.6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    bg_index = class_names.index("background") if "background" in class_names else None

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(pred_binary):
            break

        scores = pred_scores[frame_idx]  # shape: (C,)
        active = pred_binary[frame_idx]  # shape: (C,)

        top_indices = scores.argsort()[-3:][::-1]  # Top 3 confidence indices
        labels_to_show = [(class_names[i], scores[i]) for i in top_indices]

        # Optional: draw semi-transparent background
        overlay = frame.copy()
        box_height = int((25 + 5) * len(labels_to_show))
        cv2.rectangle(overlay, (5, 5), (320, 10 + box_height), (0, 0, 0), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw class labels and scores
        y = 25
        for label, conf in labels_to_show:
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 0), 1, cv2.LINE_AA)
            y += int(25 * font_scale)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f" Saved annotated video to {output_path}")

def find_best_thresholds_per_class(pred_scores, true_labels, thresholds=np.linspace(0.01, 0.99, 50), verbose=True, class_names=None):
    num_classes = pred_scores.shape[1]
    best_thresholds = []
    best_f1s = []

    for c in range(num_classes):
        y_true = true_labels[:, c]
        y_pred_scores = pred_scores[:, c]

        best_f1 = 0
        best_thresh = 0.5

        for t in thresholds:
            y_pred_bin = (y_pred_scores >= t).astype(int)
            if y_true.sum() == 0 and y_pred_bin.sum() == 0:
                f1 = 1.0  # Perfect absence match
            else:
                f1 = f1_score(y_true, y_pred_bin, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        best_thresholds.append(best_thresh)
        best_f1s.append(best_f1)

        if verbose:
            cname = class_names[c] if class_names else f"Class {c}"
            print(f"{c:2d} | {cname:30s} | Best Threshold: {best_thresh:.3f} | F1: {best_f1:.4f}")

    return best_thresholds, best_f1s

def make_binary_predictions(scores, decision_type="threshold", thresholds=None, top_n=2):
    T, C = scores.shape
    binary_preds = np.zeros_like(scores, dtype=int)

    if decision_type == "threshold":
        if thresholds is None:
            raise ValueError("Thresholds must be provided for 'threshold' decision type.")
        thresholds = np.array(thresholds).reshape(1, -1)  # shape (1, C)
        binary_preds = (scores >= thresholds).astype(int)

    elif decision_type == "top-n":
        top_indices = np.argsort(scores, axis=1)[:, -top_n:]  # shape (T, top_n)
        for i in range(T):
            binary_preds[i, top_indices[i]] = 1

    elif decision_type == "argmax":
        max_indices = scores.argmax(axis=1)  # shape (T,)
        for i in range(T):
            binary_preds[i, max_indices[i]] = 1

    else:
        raise ValueError(f"Unsupported decision_type '{decision_type}'. Use 'threshold', 'top-n', or 'argmax'.")

    return binary_preds


def end_to_end(video_path, model_path, pickle_path, model_name="MCG-NJU/videomae-large", outputpath="annotated_video.mp4"):
   
  with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

  labels = data['labels'] 
  labels.append("background")
  model = VideoMAEActionDetector(num_classes=len(labels), pretrained_model_name=model_name)
  state_dict = torch.load(model_path, map_location="cpu")
  model.load_state_dict(state_dict)
  model.eval()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  pred_scores = infer_video_fast(model=model, video_path=video_path, num_classes=len(labels))

  binary_preds = make_binary_predictions(scores=pred_scores, decision_type="threshold", thresholds=THRESHOLDS, top_n=3)

  overlay_labels_on_video(
  video_path=video_path,
  pred_binary=binary_preds,
  pred_scores=pred_scores,
  class_names=labels,
  output_path=outputpath)

  return pred_scores