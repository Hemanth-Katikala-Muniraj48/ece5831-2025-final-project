# infer_on_video_bilstm.py| OG
# ------------------------------------------------------------
# Inference with BiLSTM sequence model on a saved video.
# - Predicts: bench / sit_ups / squat
# - Counts REPS ONLY for squats (knee-angle thresholds with hysteresis)
# - Overlays: "pred: <label> | Reps: <n>"
# ------------------------------------------------------------

import os, json, cv2, joblib, numpy as np, torch
import mediapipe as mp
import torch.nn as nn

# =================== CONFIG ===================
# Paths to your trained artifacts
MODEL_PATH   = r"runs/train_bilstm/best.pt"        # <- change if needed
SCALER_PATH  = r"runs/train_bilstm/scaler.joblib"
LABELS_PATH  = r"runs/train_bilstm/labels.json"

# Input/Output
INPUT_VIDEO  = r"C:\Users\DELL\PycharmProjects\Pattern_Recognition\videos\Combined2.mp4"  # <- change
WRITE_OUTPUT = True
OUTPUT_VIDEO = "annotated_output.mp4"
SHOW_WINDOW  = True

# Smoothing / display
PRED_SMOOTH_N = 5       # majority vote over last N predictions
TEXT_POS      = (10, 36)
TEXT_SCALE    = 1.0
TEXT_THICK    = 2

# Squat rep thresholds (degrees)
KNEE_DOWN_DEG = 95.0    # below = down phase
KNEE_UP_DEG   = 160.0   # above = count a rep (up)
# ==============================================


# ---------- Model must match training ----------
class BiLSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, bidirectional=True, dropout=0.5, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        feat = out[:, -1, :]   # last timestep
        return self.head(feat)


# ---------- Pose/feature utilities ----------
FEATURE_COLS = [
    "lknee","rknee","lhip","rhip","lelbow","relbow",
    "lshoulder","rshoulder","hip_y","wrist_y"
]

def angle(a, b, c, eps=1e-6):
    ab, cb = a - b, c - b
    den = (np.linalg.norm(ab) * np.linalg.norm(cb)) + eps
    cosang = np.dot(ab, cb) / den
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def get_xy(lm, idx, w, h):
    p = lm[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)

def extract_features_from_landmarks(lm, w, h):
    # Landmarks indices (MediaPipe BlazePose full-body)
    idx = dict(
        L_SH=11, R_SH=12, L_EL=13, R_EL=14, L_WR=15, R_WR=16,
        L_HI=23, R_HI=24, L_KN=25, R_KN=26, L_AN=27, R_AN=28
    )
    P = {k: get_xy(lm, v, w, h) for k, v in idx.items()}
    sw = float(np.linalg.norm(P["R_SH"] - P["L_SH"]) + 1e-6)

    feats = np.array([
        angle(P["L_HI"], P["L_KN"], P["L_AN"]),                        # lknee
        angle(P["R_HI"], P["R_KN"], P["R_AN"]),                        # rknee
        angle(P["L_SH"], P["L_HI"], P["L_KN"]),                        # lhip
        angle(P["R_SH"], P["R_HI"], P["R_KN"]),                        # rhip
        angle(P["L_SH"], P["L_EL"], P["L_WR"]),                        # lelbow
        angle(P["R_SH"], P["R_EL"], P["R_WR"]),                        # relbow
        angle(P["L_EL"], P["L_SH"], P["L_HI"]),                        # lshoulder
        angle(P["R_EL"], P["R_SH"], P["R_HI"]),                        # rshoulder
        ((P["L_HI"][1] + P["R_HI"][1]) / 2.0 - (P["L_SH"][1] + P["R_SH"][1]) / 2.0) / sw,  # hip_y
        ((P["L_WR"][1] + P["R_WR"][1]) / 2.0 - (P["L_SH"][1] + P["R_SH"][1]) / 2.0) / sw,  # wrist_y
    ], dtype=np.float32)

    # Additional convenience for rep logic
    left_knee  = float(feats[0])
    right_knee = float(feats[1])
    min_knee   = min(left_knee, right_knee)
    return feats, min_knee


def majority_label(pred_buffer):
    if not pred_buffer:
        return "unknown"
    vals, counts = np.unique(np.array(pred_buffer), return_counts=True)
    return vals[counts.argmax()]


def main():
    # ----- Load artifacts -----
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(SCALER_PATH)
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(LABELS_PATH)

    ckpt   = torch.load(MODEL_PATH, map_location="cpu")
    config = ckpt.get("config", {})
    labels = ckpt.get("labels", None)
    if labels is None and os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            labels = json.load(f)
    if labels is None:
        raise RuntimeError("Could not load class labels.")

    window_size = int(config.get("window_size", 60))
    in_dim      = int(config.get("in_dim", len(FEATURE_COLS)))
    hidden      = int(config.get("hidden", 128))
    layers      = int(config.get("layers", 2))
    bidirectional = bool(config.get("bidirectional", True))
    dropout     = float(config.get("dropout", 0.5))
    num_classes = int(config.get("num_classes", len(labels)))

    model = BiLSTMClassifier(
        in_dim=in_dim, hidden=hidden, layers=layers,
        bidirectional=bidirectional, dropout=dropout,
        num_classes=num_classes
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = joblib.load(SCALER_PATH)

    # ----- Video IO -----
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if WRITE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))

    # ----- Pose + buffers -----
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    mp_conn = mp.solutions.pose.POSE_CONNECTIONS

    seq_buffer   = []   # holds scaled features (for LSTM)
    pred_buffer  = []   # last N predicted labels for smoothing
    reps         = 0
    down_phase   = False  # for squat hysteresis only

    print(f"[INFO] classes={labels} | window={window_size} | in_dim={in_dim}")

    with mp_pose.Pose(model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]

            # Run pose
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            pred_label = "unknown"
            knee_min   = None

            if res.pose_landmarks:
                # Extract features
                feats, knee_min = extract_features_from_landmarks(res.pose_landmarks.landmark, w, h)

                # Scale features using training scaler
                feats_scaled = scaler.transform(feats.reshape(1, -1)).reshape(-1).astype(np.float32)

                # Build sequence window
                seq_buffer.append(feats_scaled)
                if len(seq_buffer) > window_size:
                    seq_buffer.pop(0)

                # Predict when we have enough frames
                if len(seq_buffer) == window_size:
                    x = torch.from_numpy(np.stack(seq_buffer, axis=0)).unsqueeze(0)  # (1,T,F)
                    with torch.no_grad():
                        logits = model(x)
                        pred_idx = int(torch.argmax(logits, dim=1).item())
                    pred_label = labels[pred_idx]
                    # Smooth prediction
                    pred_buffer.append(pred_label)
                    if len(pred_buffer) > PRED_SMOOTH_N:
                        pred_buffer.pop(0)
                    pred_label = majority_label(pred_buffer)

                # Draw skeleton (optional)
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_conn)

            # ---- Rep counting for SQUATS only ----
            if pred_label == "squat" and knee_min is not None:
                # go down
                if knee_min < KNEE_DOWN_DEG and not down_phase:
                    down_phase = True
                # come up -> count
                if knee_min > KNEE_UP_DEG and down_phase:
                    reps += 1
                    down_phase = False
            else:
                # if not squat, don't update down_phase to avoid spurious counts
                pass

            # ---- Overlay text ----
            overlay = f"pred: {pred_label} | Reps: {reps}"
            cv2.putText(frame, overlay, TEXT_POS, cv2.FONT_HERSHEY_SIMPLEX,
                        TEXT_SCALE, (0, 255, 0), TEXT_THICK, cv2.LINE_AA)

            # Write / show
            if WRITE_OUTPUT:
                writer.write(frame)
            if SHOW_WINDOW:
                cv2.imshow("BiLSTM Inference", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break

    cap.release()
    if writer:
        writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    if WRITE_OUTPUT:
        print(f"[OK] saved annotated video -> {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
