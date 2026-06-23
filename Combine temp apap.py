# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 16:59:30 2026

@author: param
"""

import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import os
from datetime import datetime

# ────────────────────────────────────────────────
#               TUNABLE SMOOTHING PARAMETERS
# ────────────────────────────────────────────────
ROLLING_WINDOW = 15          # First stage: robust rolling mean
GAUSSIAN_SIGMA = 3.0         # Second stage: gentle gaussian polish (smaller = sharper)
RATE_SMOOTH_WINDOW = 21      # For rate-of-change curve (needs to be smoother)
BASELINE_END_TIME = 300.0    # Use data from 0 to 300 s for baseline
POLY_DEGREE = 5              # unchanged - for trend line
# ────────────────────────────────────────────────

class MultiExperimentPlotter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Multi-Experiment Thermal Analysis Plotter")
        
        self.groups = [{} for _ in range(5)]
        self.group_frames = []
        
        for i in range(5):
            frame = tk.Frame(self.root)
            frame.pack(pady=10)
            
            tk.Label(frame, text=f"Group {i+1}:").pack(side=tk.LEFT)
            
            name_entry = tk.Entry(frame)
            name_entry.insert(0, f"Group {i+1}")
            name_entry.pack(side=tk.LEFT, padx=5)
            self.groups[i]['name_entry'] = name_entry
            
            color_button = tk.Button(frame, text="Choose Color", command=lambda idx=i: self.choose_color(idx))
            color_button.pack(side=tk.LEFT, padx=5)
            self.groups[i]['color'] = f'C{i}'
            self.groups[i]['color_button'] = color_button
            
            files_button = tk.Button(frame, text="Select CSV Files", command=lambda idx=i: self.select_files(idx))
            files_button.pack(side=tk.LEFT, padx=5)
            self.groups[i]['files'] = []
            self.groups[i]['files_label'] = tk.Label(frame, text="0 files selected")
            self.groups[i]['files_label'].pack(side=tk.LEFT, padx=5)
            
            self.group_frames.append(frame)
        
        plot_button = tk.Button(self.root, text="Generate Plots", command=self.generate_plots)
        plot_button.pack(pady=20)
        
        self.root.mainloop()
    
    def choose_color(self, group_idx):
        color = colorchooser.askcolor()[1]
        if color:
            self.groups[group_idx]['color'] = color
            self.groups[group_idx]['color_button'].config(bg=color)
    
    def select_files(self, group_idx):
        files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if files:
            self.groups[group_idx]['files'] = list(files)
            self.groups[group_idx]['files_label'].config(text=f"{len(files)} files selected")
    
    def _smooth_series(self, y):
        """Robust two-stage smoothing with excellent edge behavior"""
        if len(y) < ROLLING_WINDOW:
            return y.copy()
        
        # Stage 1: Robust rolling mean (pandas handles edges gracefully)
        y_series = pd.Series(y)
        y_smooth = y_series.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean().values
        
        # Stage 2: Light Gaussian smoothing
        y_smooth = ndimage.gaussian_filter1d(y_smooth, sigma=GAUSSIAN_SIGMA)
        
        return y_smooth

    def generate_plots(self):
        experiment_data = {}
        min_duration = float('inf')
        
        for i, group in enumerate(self.groups):
            if not group['files']:
                continue
            
            group_name = group['name_entry'].get()
            color = group['color']
            dfs = []
            
            for file in group['files']:
                df = pd.read_csv(file)
                dfs.append(df)
                min_duration = min(min_duration, df['time_s'].max())
            
            experiment_data[group_name] = {
                'dfs': dfs,
                'color': color
            }
        
        if not experiment_data:
            messagebox.showerror("Error", "No files selected in any group.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"multi_experiment_analysis_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        common_times = np.arange(0, min_duration + 1, 30.0)
        
        self.plot_normalized_over_time(experiment_data, common_times, output_dir, 'roi_max', 'Normalized ROI Maximum Temperature')
        self.plot_normalized_over_time(experiment_data, common_times, output_dir, 'corrected_max', 'Normalized Max Pixel Temperature')
        
        self.plot_rate_of_change(experiment_data, common_times, output_dir, 'roi_max', 'Rate of Change - ROI Maximum')
        self.plot_rate_of_change(experiment_data, common_times, output_dir, 'corrected_max', 'Rate of Change - Max Pixel')
        
        messagebox.showinfo("Success", f"Plots generated in {output_dir}")
    
    def plot_normalized_over_time(self, experiment_data, common_times, output_dir, value_col, title):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for group_name, data in experiment_data.items():
            dfs = data['dfs']
            color = data['color']
            
            all_values = []
            for df in dfs:
                times = df['time_s'].values
                norm_val = df[f'normalized_{value_col}'].values
                
                # Compute baseline offset
                baseline_mask = times <= BASELINE_END_TIME
                baseline_offset = np.mean(norm_val[baseline_mask]) if np.any(baseline_mask) else 0.0
                
                norm_val_corrected = norm_val - baseline_offset
                
                # Interpolate
                df_interp = np.interp(common_times, times, norm_val_corrected)
                
                # Apply new robust smoothing
                df_smooth = self._smooth_series(df_interp)
                
                all_values.append(df_smooth)
            
            all_values = np.array(all_values)
            means = np.mean(all_values, axis=0)
            sems = np.std(all_values, axis=0) / np.sqrt(len(dfs)) if len(dfs) > 1 else np.zeros_like(means)
            
            ax.plot(common_times, means, label=group_name, color=color, linewidth=2.5)
            ax.fill_between(common_times, means - 0.5 * sems, means + 0.5 * sems, alpha=0.3, color=color)
            
            if len(common_times) > POLY_DEGREE:
                p = np.polyfit(common_times, means, POLY_DEGREE)
                fit = np.polyval(p, common_times)
                ax.plot(common_times, fit, linestyle='--', color=color, linewidth=1.5,
                        label=f'{group_name} Poly Fit')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Normalized Temperature (°C)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{value_col}_normalized_over_time.png'), dpi=300)
        plt.close()
    
    def plot_rate_of_change(self, experiment_data, common_times, output_dir, value_col, title):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for group_name, data in experiment_data.items():
            dfs = data['dfs']
            color = data['color']
            
            all_values = []
            for df in dfs:
                times = df['time_s'].values
                norm_val = df[f'normalized_{value_col}'].values
                
                baseline_mask = times <= BASELINE_END_TIME
                baseline_offset = np.mean(norm_val[baseline_mask]) if np.any(baseline_mask) else 0.0
                
                norm_val_corrected = norm_val - baseline_offset
                
                df_interp = np.interp(common_times, times, norm_val_corrected)
                
                # Same robust smoothing
                df_smooth = self._smooth_series(df_interp)
                
                all_values.append(df_smooth)
            
            all_values = np.array(all_values)
            means = np.mean(all_values, axis=0)
            
            # Compute rate (in °C/min)
            rate = np.gradient(means, common_times) * 60
            
            # Extra smoothing on the derivative (derivatives amplify noise)
            if len(rate) > RATE_SMOOTH_WINDOW:
                rate_smoothed = ndimage.gaussian_filter1d(rate, sigma=GAUSSIAN_SIGMA * 1.5)
            else:
                rate_smoothed = rate
            
            ax.plot(common_times, rate_smoothed, label=group_name, color=color, linewidth=2.5)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Rate of Change (°C/min)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{value_col}_rate_of_change.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    MultiExperimentPlotter()