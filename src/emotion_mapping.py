# src/emotion_mapping.py
import numpy as np
import pandas as pd
from datasets import load_dataset

# ---------------------------
# Helper: check valid hex
# ---------------------------
def is_valid_hex(s):
    """Check if a string is a valid hex color like '#FFAABB'."""
    if not isinstance(s, str):
        return False
    s = s.lstrip('#')
    if len(s) != 6:
        return False
    try:
        int(s, 16)
        return True
    except ValueError:
        return False

# ---------------------------
# Load dataset safely
# ---------------------------
def load_color_dataset():
    """
    Loads the boltuix/color-pedia dataset and converts hex colors to RGB.
    Returns a DataFrame with columns ['R','G','B','emotion'].
    """
    ds = load_dataset("boltuix/color-pedia")
    df = pd.DataFrame(ds['train'])

    # Detect hex column automatically
    hex_col_candidates = [c for c in df.columns if 'hex' in c.lower() or 'color' in c.lower()]
    if not hex_col_candidates:
        raise ValueError("No hex/color column found in dataset")
    hex_col = hex_col_candidates[0]

    # Pick a column for emotion / tags
    emotion_col_candidates = [c for c in df.columns if 'tag' in c.lower() or 'description' in c.lower()]
    if not emotion_col_candidates:
        raise ValueError("No column found for emotion/tags in dataset")
    emotion_col = emotion_col_candidates[0]

    # Filter only valid hex rows
    df = df[df[hex_col].apply(is_valid_hex)].copy()

    # Convert hex to RGB
    df[['R','G','B']] = df[hex_col].apply(lambda x: pd.Series(
        tuple(int(x.lstrip('#')[i:i+2], 16) for i in (0,2,4))
    ))

    # Keep only necessary columns and rename emotion column
    df_colors = df[['R','G','B', emotion_col]].rename(columns={emotion_col:'emotion'}).reset_index(drop=True)
    return df_colors

# Load dataset once
df_colors = load_color_dataset()
print(f"[INFO] Loaded color-pedia dataset with {len(df_colors)} valid colors.")

# ---------------------------
# Map RGB to closest emotion
# ---------------------------
def color_to_emotion(rgb):
    """
    Maps an RGB color to the closest color in the color-pedia dataset.

    Args:
        rgb: tuple of (R,G,B) values (0-255)

    Returns:
        emotion: string
    """
    distances = np.sqrt((df_colors['R'] - rgb[0])**2 +
                        (df_colors['G'] - rgb[1])**2 +
                        (df_colors['B'] - rgb[2])**2)
    idx = distances.idxmin()
    return df_colors.loc[idx, 'emotion']
