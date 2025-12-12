# extract_frames_to_ranges_csv.py
# ------------------------------------------------------------
# Builds ranges_dataset_grouped.csv from per-frame folders.
# Directory layout expected:
#   ROOT/
#     bench/
#       0341/  <frames>
#       0342/  <frames>
#       ...
#     sit_ups/
#       1559/  <frames>
#       ...
#     squat/
#       1659/  <frames>
#       ...
#
# Output CSV columns:
#   group_id, frame_idx, label, lknee, rknee, lhip, rhip,
#   lelbow, relbow, lshoulder, rshoulder, hip_y, wrist_y
# ------------------------------------------------------------

import os, glob, csv, re
import cv2
import numpy as np
import mediapipe as mp

# --------------- EDIT THESE ---------------
ROOT    = r"D:\Dataset"                # <-- your root folder
OUT_CSV = "ranges_dataset_grouped.csv"
CLASSES = ["bench", "sit_ups", "squat"]  # subfolder names under ROOT
# ------------------------------------------

FEATURE_COLS = [
    "lknee","rknee","lhip","rhip","lelbow","relbow",
    "lshoulder","rshoulder","hip_y","wrist_y"
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def natural_key(s):
    """Sort filenames by human/numeric order (e.g., 9 < 10, 099 < 100)."""
    s = os.path.basename(s)
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]

def list_frames(folder):
    """Return sorted list of frame image paths."""
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    files.sort(key=natural_key)
    return files

def angle(a, b, c, eps=1e-6):
    ab, cb = a - b, c - b
    den = (np.linalg.norm(ab) * np.linalg.norm(cb)) + eps
    cosang = np.dot(ab, cb) / den
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def get_xy(lms, idx, w, h):
    p = lms[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)

def process_group(group_path, label, writer, show_every=0):
    """Process one leaf folder (group_id) containing frames."""
    group_id = os.path.basename(group_path.rstrip("\\/"))
    frames = list_frames(group_path)
    if not frames:
        print(f"[WARN] No frames in {group_path}")
        return 0

    mp_pose = mp.solutions.pose
    nrows = 0
    with mp_pose.Pose(model_complexity=1, enable_segmentation=False) as pose:
        for frame_idx, img_path in enumerate(frames):
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            idx = dict(
                L_SH=11, R_SH=12, L_EL=13, R_EL=14, L_WR=15, R_WR=16,
                L_HI=23, R_HI=24, L_KN=25, R_KN=26, L_AN=27, R_AN=28
            )
            P = {k: get_xy(lm, v, w, h) for k, v in idx.items()}
            sw = float(np.linalg.norm(P["R_SH"] - P["L_SH"]) + 1e-6)

            feats = [
                angle(P["L_HI"], P["L_KN"], P["L_AN"]),  # lknee
                angle(P["R_HI"], P["R_KN"], P["R_AN"]),  # rknee
                angle(P["L_SH"], P["L_HI"], P["L_KN"]),  # lhip
                angle(P["R_SH"], P["R_HI"], P["R_KN"]),  # rhip
                angle(P["L_SH"], P["L_EL"], P["L_WR"]),  # lelbow
                angle(P["R_SH"], P["R_EL"], P["R_WR"]),  # relbow
                angle(P["L_EL"], P["L_SH"], P["L_HI"]),  # lshoulder
                angle(P["R_EL"], P["R_SH"], P["R_HI"]),  # rshoulder
                ((P["L_HI"][1]+P["R_HI"][1])/2.0 - (P["L_SH"][1]+P["R_SH"][1])/2.0)/sw,  # hip_y
                ((P["L_WR"][1]+P["R_WR"][1])/2.0 - (P["L_SH"][1]+P["R_SH"][1])/2.0)/sw,  # wrist_y
            ]
            row = [group_id, frame_idx, label] + [float(x) for x in feats]
            writer.writerow(row)
            nrows += 1

    print(f"[OK] {label}/{group_id}: {nrows} rows")
    return nrows

def main():
    groups = []
    for cls in CLASSES:
        cls_dir = os.path.join(ROOT, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Missing class folder: {cls_dir}")
            continue
        # each subfolder (e.g., 0341, 0342, ...) is a group
        for group in sorted(os.listdir(cls_dir)):
            gpath = os.path.join(cls_dir, group)
            if os.path.isdir(gpath):
                groups.append((gpath, cls))

    if not groups:
        print(f"[ERR] No groups found under {ROOT} with classes {CLASSES}")
        return

    total = 0
    with open(OUT_CSV, "w", newline="") as fp:
        w = csv.writer(fp)
        header = ["group_id","frame_idx","label"] + FEATURE_COLS
        w.writerow(header)
        for gpath, lbl in groups:
            total += process_group(gpath, lbl, w)

    print(f"[DONE] wrote {OUT_CSV} with {total} rows total.")

if __name__ == "__main__":
    main()
