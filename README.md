# Real‑Time Pose‑Based Exercise Recognition with Bi‑LSTM

> **ECE 5831 — Pattern Recognition & Neural Networks**  
> **Student:** Hemanth Katikala Muniraj  
> **Classes:** `bench`, `sit_ups`, `squat`  
> **Goal:** Real‑time recognition of exercise type from webcam/video using MediaPipe Pose + pose‑feature + a 2‑layer Bi‑LSTM. Includes a **squat‑only rep counter** based on knee‑angle hysteresis.

---

## 📺 Deliverables

- **Demo video (YouTube):** `TODO: https://youtu.be/...`
- **Slide deck (Google Drive):** `TODO: https://drive.google.com/...`
- **Report (IEEE PDF):** `TODO: https://drive.google.com/...`
- **Dataset (Drive or instructions):** `TODO: https://drive.google.com/...`
- **Interactive notebook:** [`final-project.ipynb`](./final-project.ipynb)
- **Annotated demo output:** `annotated_output.mp4` (created by the notebook or inference script)

---

## 🧭 Repository Structure

```
.
├─ final-project.ipynb                 # Minimal demo: loads best.pt, runs inference, saves annotated MP4
├─ train_sequence_nn.py                # Train 2×Bi‑LSTM on windowed pose features → best.pt, scaler, labels, results
├─ infer_on_video_bilstm.py            # Robust video inference + squat‑only reps + bench plausibility checks
├─ extract_ranges_to_csv_from_frames.py# Build ranges_dataset.csv from frame folders (bench/sit_ups/squat)
├─ csv_grouped.py           # (Optional) Build ranges_dataset_grouped.csv from Penn Action .mat labels
├─ runs/
│   └─ train_bilstm/
│       ├─ best.pt                     # Trained checkpoint (with config + labels)
│       ├─ scaler.joblib               # sklearn StandardScaler used at train time
│       ├─ labels.json                 # ["bench","sit_ups","squat"]
│       ├─ results.csv                 # Per‑epoch metrics
│       ├─ results.png                 # YOLO‑style plots (loss/acc + best epoch marker)
│       ├─ confmat_val.png             # Validation confusion matrix
│       └─ confmat_test.png            # Test confusion matrix (if generated)
└─ README.md
```

---

## 🧩 System Overview

1. **Pose extraction (MediaPipe Pose)** → 33 landmarks per frame.  
2. **Feature engineering** → 8 joint angles (knees/hips/elbows/shoulders) + 2 normalized distances (`hip_y`, `wrist_y` normalized by shoulder width).  
3. **Temporal modeling** → 60‑frame sliding windows (~2 s at 30 FPS) into a **2‑layer Bi‑LSTM (128 units/layer) + Dense(256)+Dropout**.  
4. **Inference** → Live/video: pose → features → sliding window → predicted class (`bench | sit_ups | squat`).  
5. **Rep counting** → **Squats only**, using knee‑angle thresholds (down < ~95°, up > ~160°) with hysteresis.  
6. **Bench robustness** → Upscaling, ROI retry, EMA landmark smoothing, short‑gap feature reuse, and **bench plausibility checks** reduce “unknown/flicker” when wrists are confused with ankles.

---

## 📦 Requirements

- Python **3.10–3.11** recommended
- Packages:
  - `torch` (CPU or CUDA)
  - `mediapipe`
  - `opencv-python`
  - `numpy`, `pandas`
  - `scikit-learn` (for `StandardScaler` deserialization)
  - `joblib`
  - `matplotlib`

**Quick install (pip):**
```bash
python -m venv .venv
# Windows:
. .venv\\Scripts\\activate
# macOS/Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install torch mediapipe opencv-python numpy pandas scikit-learn==1.3.2 joblib matplotlib
```

---

## 📂 Data Preparation

Two supported layouts.

### A) Frame folders by class (recommended for quick start)
```
D:/Dataset/
 ├─ bench/
 │   ├─ 0341/    # frames: 00001.jpg, 00002.jpg, ...
 │   ├─ 0342/ 
 │   └─ 0480/
 ├─ sit_ups/
 │   ├─ 1559/ ... 1658/
 └─ squat/
     ├─ 1659/ ... 1889/
```

Generate `ranges_dataset.csv` from frames:
```bash
python extract_ranges_to_csv_from_frames.py --root "D:/Dataset" --out "ranges_dataset.csv"
```

Columns include: `lknee, rknee, lhip, rhip, lelbow, relbow, lshoulder, rshoulder, hip_y, wrist_y, label, group_id, frame_idx`.

### B) From Penn Action label .mat files
If you have `Penn_Action/labels/*.mat`, build a leakage‑safe index:
```bash
python csv_grouped.py   # writes penn_labels_index.csv and prints targets
# then run your extractor to produce ranges_dataset_grouped.csv
```

> **Important:** Split **by `group_id` (session)** *before* windowing to avoid train/val/test leakage.

---

## 🏋️ Training the Bi‑LSTM

```bash
python train_sequence_nn.py --csv_path ranges_dataset.csv --epochs 50 --batch_size 64
```

**Default hyperparameters** (also saved into `best.pt → config`):
- `WINDOW_SIZE=60`, `HOP_TRAIN=30`, `HOP_VAL=30`
- BiLSTM: `layers=2`, `hidden=128`, `bidirectional=True`
- Dense head `256` + `Dropout(0.5)`
- `AdamW`, `weight_decay=1e-4`, `label_smoothing=0.05`, gradient clip `5.0`
- `ReduceLROnPlateau` on val accuracy, early stop on patience

**Outputs** in `runs/train_bilstm/`:
- `best.pt`, `scaler.joblib`, `labels.json`
- `results.csv`, `results.png`, `confmat_*.png`
- Console prints: **validation accuracy**, **per‑class P/R/F1**, **confusion matrix**

---

## 🎬 Inference on a Video (robust, with squat reps)

Edit paths inside `infer_on_video_bilstm.py`, then run:
```bash
python infer_on_video_bilstm.py
```
- Overlays: `pred: <label> | Reps: <n>`  
- Writes `annotated_output.mp4`  
- Bench‑stability features:
  - Upscale for pose if frame < `720px` tall (`MIN_INFER_H`)
  - EMA landmark smoothing (`EMA_ALPHA`)
  - Retry pose on torso ROI (`ROI_GROW`)
  - Short‑gap feature reuse (`MISS_TOL`)
  - **Bench plausibility checks** to reject impossible wrist/shoulder geometry

---

## 🧪 Minimal Demo Notebook

Open **[`final-project.ipynb`](./final-project.ipynb)** and edit:
```python
MODEL_PATH   = r"C:/Users/DELL/Documents/ece5831-2025-assignments/ece5831-2025-final-project/runs/train_bilstm/best.pt"
SCALER_PATH  = r"C:/Users/DELL/Documents/ece5831-2025-assignments/ece5831-2025-final-project/runs/train_bilstm/scaler.joblib"
LABELS_PATH  = r"C:/Users/DELL/Documents/ece5831-2025-assignments/ece5831-2025-final-project/runs/train_bilstm/labels.json"
INPUT_VIDEO  = r"C:/Users/DELL/Documents/ece5831-2025-assignments/ece5831-2025-final-project/Videos/Squat.mp4"

```
Run all cells to:
- Load the trained model + scaler + labels
- Run MediaPipe Pose + Bi‑LSTM on a short clip
- Save **`annotated_output.mp4`**
- Show a few sampled frames inline

If you see `ModuleNotFoundError: sklearn`, run in a new cell:
```python
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.3.2", "joblib"])
```

---

## 📊 Current Validation Results

- **Val Acc:** `0.9812` (best epoch: 36)

Per‑class P/R/F1 (support):
- **bench:** 0.980 / 0.943 / 0.962 (n=159)  
- **sit_ups:** 0.972 / 0.993 / 0.983 (n=142)  
- **squat:** 0.985 / 0.994 / 0.990 (n=337)

*Splits are by `group_id` (session), not by frame.*

---

## 🩹 Troubleshooting

- **`ModuleNotFoundError: sklearn`** when loading `scaler.joblib`  
  Install `scikit-learn==1.3.2` and re‑run.

- **`Cannot open video`**  
  Check `INPUT_VIDEO` path and codec; try `mp4v` writer; ensure file is accessible.

- **Pose misses wrists/arms on bench**  
  Increase `MIN_INFER_H` to `900`, set `POSE_DET_CONF=0.8`, `POSE_TRK_CONF=0.8`, increase `PRED_SMOOTH_N` to `9–11`.  
  Prefer higher‑contrast clothing and a perpendicular camera angle.

- **Validation too high**  
  Ensure `ranges_dataset*.csv` is split **by `group_id` before windowing**.

---

## 🔗 Submission Links (fill these in)

- **YouTube demo:** `TODO`
- **Slides (Drive):** `TODO`
- **Report (IEEE PDF, Drive):** `TODO`
- **Dataset (Drive):** `TODO`

---

## ✅ Submission Checklist

- [ ] Repo name is **`ece-5831-2025-final-project`** and is **Public**  
- [ ] `final-project.ipynb` runs end‑to‑end and shows **executed cells**  
- [ ] `README.md` contains links to **demo video**, **slides**, and **report**  
- [ ] **Google Drive** folder `ece-5831-2025-final-project` with `dataset/`, `presentation/`, `report` is shared “Anyone with the link”  
- [ ] **YouTube** demo uploaded and linked  
- [ ] `runs/train_bilstm/` includes `best.pt`, `scaler.joblib`, `labels.json`, `results.csv`, plots, `confmat_*.png`

---

## 📚 References / Acknowledgments

- MediaPipe Pose — Google Research  
- Penn Action Dataset — Zhang et al., UPenn  
- PyTorch, scikit‑learn, OpenCV, NumPy, Matplotlib  

*(Full citations will be included in the IEEE report.)*

---

## 🔒 License & Academic Use

This repository is for educational purposes for **ECE 5831**. If redistributing dataset frames, ensure compliance with the original dataset license/terms. For Penn Action, link to the source or provide your own captured data.
