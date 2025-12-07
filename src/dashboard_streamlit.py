# src/dashboard_streamlit.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from src.preprocessing import load_image_as_rgb, pil_from_array
from src.color_extraction import extract_dominant_colors
from src.emotion_mapping import color_to_emotion
from src.analytics import create_summary_rows, save_summary_df
from src.visualization import (
    rgb_to_hex, plot_color_strip_hex, plot_percentage_bar,
    plot_pie_emotions, plot_rgb_hist, plot_emotion_stacked_bar, plot_emotion_heatmap
)
import io
from tqdm import tqdm

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Color Mood Analytics", layout="wide")

BASE_DIR = Path.cwd()
DATA_IN = BASE_DIR / "data" / "input" / "user_uploads"
OUT_DIR = BASE_DIR / "data" / "output"

DATA_IN.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

st.title("🖼️ City Color Analytics — Live Dashboard")
st.markdown(
    "Upload images and get dominant color palettes + emotion visualizations. "
    "This is a Data Visualization project focusing on color trends and mood analysis."
)

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Options")
k = st.sidebar.slider("Number of dominant colors (k)", min_value=3, max_value=8, value=5)
resize = st.sidebar.slider("Resize max pixels (for speed)", 200, 1600, 800)
sample_frac = st.sidebar.slider("Sampling fraction for clustering (speed)", 0.1, 1.0, 1.0)
show_aggregates = st.sidebar.checkbox("Show aggregated visuals across images", value=True)

# -----------------------
# Upload files
# -----------------------
uploaded_files = st.file_uploader("Upload one or more images", type=['png','jpg','jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.sidebar.info(f"{len(uploaded_files)} files uploaded")
    all_rows = []
    progress = st.progress(0)
    
    # Save uploaded files
    for f in uploaded_files:
        out_path = DATA_IN / f.name
        with open(out_path, "wb") as wf:
            wf.write(f.getbuffer())

    # Process each uploaded image
    for i, f in enumerate(uploaded_files):
        st.header(f"Image: {f.name}")
        img_np = load_image_as_rgb(DATA_IN / f.name, resize_max=resize)

        # Extract dominant colors
        centers, fractions = extract_dominant_colors(img_np, k=k, sample_frac=sample_frac)
        centers = np.array(centers)
        fractions = np.array(fractions)

        # Convert to hex & map emotions
        hexes = [rgb_to_hex(c) for c in centers]
        emotions = [color_to_emotion(tuple(c)) for c in centers]
        color_map = {emo: hexes[i] for i, emo in enumerate(emotions)}

        # -----------------------
        # Display original image
        # -----------------------
        st.image(f, caption=f.name, use_column_width='always')

        # -----------------------
        # Layout: Color Strip, Top Colors, Emotion Distribution
        # -----------------------
        col1, col2, col3 = st.columns([1,1,1])
        
        with col1:
            st.subheader("Color Strip")
            buf = plot_color_strip_hex(hexes, fractions, title="Color Strip")
            st.image(buf)

        with col2:
            st.subheader("Top Colors (%)")
            buf2 = plot_percentage_bar(hexes, fractions, title="Top Colors")
            st.image(buf2)

        with col3:
            st.subheader("Emotion Distribution")
            emo_counts = {}
            for emo, frac in zip(emotions, fractions):
                emo_counts[emo] = emo_counts.get(emo, 0) + float(frac)
            buf3 = plot_pie_emotions(emo_counts, color_map=color_map, title="Emotions")
            st.image(buf3)

        # -----------------------
        # RGB Histogram
        # -----------------------
        st.subheader("RGB Histogram")
        buf4 = plot_rgb_hist(img_np, title="RGB Histogram")
        st.image(buf4)

        # -----------------------
        # Add summary rows
        # -----------------------
        rows = create_summary_rows(f.name, centers, fractions, emotions)
        all_rows.extend(rows)

        progress.progress((i+1)/len(uploaded_files))

    # -----------------------
    # Save and show summary
    # -----------------------
    summary_path = OUT_DIR / "summary.csv"
    df = save_summary_df(all_rows, summary_path)
    st.success(f"Saved summary to {summary_path}")
    st.dataframe(df)

    # -----------------------
    # Aggregated visuals
    # -----------------------
    if show_aggregates:
        st.header("Aggregated Visuals Across Images")
        agg_df = df.groupby(['image','emotion'])['fraction'].sum().reset_index()

        st.subheader("Emotion Stacked Bar")
        buf = plot_emotion_stacked_bar(agg_df, color_map=color_map)
        st.image(buf)

        st.subheader("Emotion Heatmap")
        buf = plot_emotion_heatmap(agg_df, color_map=color_map)
        st.image(buf)

    # -----------------------
    # Download CSV
    # -----------------------
    st.download_button("Download summary CSV", data=open(summary_path,'rb'), file_name="summary.csv")
else:
    st.info("Upload images to begin. You can drag-and-drop multiple at once.")
    st.markdown("**Tips:** Use the sidebar to change number of colors (k), resizing, and sampling for speed.")
