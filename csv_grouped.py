# add_group_id_from_path.py
import os, sys
import pandas as pd

CSV_IN  = "ranges_dataset.csv"
CSV_OUT = "ranges_dataset_grouped.csv"

# Change if your column has a different name
PATH_COL_CANDIDATES = ["image_path", "frame_path", "source", "path"]

def infer_group_id(path):
    """
    Examples turned into group_id:
      D:/Penn/bench/vid_0001/frame_000123.jpg -> vid_0001
      D:/Penn/squat/s01_c02.mp4:000123       -> s01_c02
      C:/.../Bench_Press7.mp4:0123           -> Bench_Press7
    """
    p = str(path)
    base = os.path.basename(p)
    root, ext = os.path.splitext(base)
    # If frames live under per-video folders, use parent folder name:
    parent = os.path.basename(os.path.dirname(p))
    # Heuristic: if parent looks like a video-id folder, prefer it
    if any(k in parent.lower() for k in ["vid", "video", "seq", "s0", "bench", "squat", "sit_ups"]):
        return parent
    # Otherwise use file stem before any colon/index
    return root.split(":")[0]

def main():
    df = pd.read_csv(CSV_IN)
    # find the path column
    path_col = None
    for c in PATH_COL_CANDIDATES:
        if c in df.columns:
            path_col = c
            break
    if path_col is None:
        raise SystemExit(f"Could not find a path column in {PATH_COL_CANDIDATES}. "
                         f"If your CSV has no file paths, use Option B to rebuild.")
    df["group_id"] = df[path_col].apply(infer_group_id)
    df.to_csv(CSV_OUT, index=False)
    print(f"[OK] wrote {CSV_OUT} with group_id. Preview:")
    print(df[[path_col, "group_id", "label"]].head())

if __name__ == "__main__":
    main()
