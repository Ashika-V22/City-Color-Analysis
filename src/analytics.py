# src/analytics.py
import pandas as pd
from pathlib import Path

def create_summary_rows(image_name, centers, fractions, emotions):
    """
    Build list of dicts for summary CSV / dataframe
    """
    rows = []
    for i, (c, f, e) in enumerate(zip(centers, fractions, emotions), start=1):
        rows.append({
            "image": image_name,
            "rank": i,
            "r": int(c[0]),
            "g": int(c[1]),
            "b": int(c[2]),
            "hex": '#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2])),
            "fraction": float(f),
            "emotion": e
        })
    return rows

def save_summary_df(rows, out_path):
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
