# src/visualization.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from PIL import Image
from io import BytesIO

sns.set()  # use seaborn style (not required)

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def plot_color_strip_hex(hex_list, frac_list, title=None, figsize=(6,1.2)):
    fig, ax = plt.subplots(figsize=figsize)
    left = 0.0
    for h, f in zip(hex_list, frac_list):
        ax.add_patch(mpatches.Rectangle((left, 0), f, 1, color=h))
        left += f
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9)
    buf = BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_percentage_bar(hex_list, frac_list, title=None, figsize=(6,3)):
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(hex_list)), frac_list)
    ax.set_xticks(range(len(hex_list)))
    ax.set_xticklabels([f'#{i+1}' for i in range(len(hex_list))])
    ax.set_ylabel('Fraction')
    if title:
        ax.set_title(title)
    for bar, hexc, frac in zip(bars, hex_list, frac_list):
        bar.set_color(hexc)
        ax.text(bar.get_x() + bar.get_width()/2, frac + 0.01, f'{frac*100:.1f}%', ha='center', fontsize=8)
    buf = BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_pie_emotions(emotion_counts:dict, color_map=None, title=None, figsize=(4,4)):
    labels = list(emotion_counts.keys())
    sizes = list(emotion_counts.values())
    colors = [color_map.get(label, "#cccccc") for label in labels] if color_map else None

    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize':8})
    ax.axis('equal')
    if title:
        ax.set_title(title)
    buf = BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_rgb_hist(image_np, title=None, figsize=(6,3)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = ('r','g','b')
    for i, col in enumerate(colors):
        ax.hist(image_np[:,:,i].ravel(), bins=256, alpha=0.6, label=col)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Pixel count')
    if title:
        ax.set_title(title)
    ax.legend()
    buf = BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_emotion_stacked_bar(df_summary, color_map=None, figsize=(10,4)):
    # df_summary: pandas DataFrame with columns ['image','emotion','fraction'] aggregated
    import pandas as pd
    pivot = df_summary.pivot_table(index='image', columns='emotion', values='fraction', aggfunc='sum', fill_value=0)
    fig, ax = plt.subplots(figsize=figsize)
    colors = [color_map.get(col, "#cccccc") for col in pivot.columns] if color_map else None
    pivot.plot(kind='bar', stacked=True, ax=ax, color=colors)
    ax.set_ylabel('Fraction')
    ax.set_title('Emotion distribution across images')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    buf = BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_emotion_heatmap(df_summary, color_map=None, figsize=(10,4)):
    import pandas as pd
    pivot = df_summary.pivot_table(index='image', columns='emotion', values='fraction', aggfunc='sum', fill_value=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    if color_map:
        # Seaborn heatmap doesn't natively support per-column colors, so use a custom palette for annotations
        cmap_colors = [color_map.get(col, "#cccccc") for col in pivot.columns]
        sns.heatmap(pivot, annot=True, fmt=".2f", cbar=False, linewidths=0.5, linecolor='gray', ax=ax, cmap=sns.color_palette(cmap_colors))
    else:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap='YlGnBu', ax=ax)
    ax.set_title('Emotion heatmap (images vs emotions)')
    
    buf = BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf
