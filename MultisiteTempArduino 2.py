# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 09:38:13 2026

@author: param

COMPLETE Thermal Analysis for Mouse Neuroscience with Arduino MLX90621
FOCUSED ON BRIGHTEST PIXEL & ROI TEMPERATURES WITH 3D VISUALIZATIONS
Always plots Max Pixel and ROI temperatures (not means) in all time series
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import time
from scipy import signal, ndimage, stats
from datetime import datetime
import warnings
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings('ignore')

# =============================================================================
# CORRECTION PARAMETERS - CORRECTED FOR ARDUINO'S NEGATIVE OFFSET
# =============================================================================
MOUSE_EMISSIVITY = 0.93        # Actual mouse emissivity (0.92-0.97)
SENSOR_EMISSIVITY = 1.0        # What MLX90621 assumes (from your code)
ARDUINO_OFFSET_IS_NEGATIVE = True  # CRITICAL: Your Arduino REDUCES temperatures
APPLY_EMISSIVITY_CORRECTION = True  # MUST BE TRUE for accurate mouse temps

# Processing parameters
SAMPLING_RATE = 30.0          # From your Arduino: RECORD_INTERVAL = 30000ms
APPLY_DELTA_T = True          # Remove ambient drift (recommended)
SMOOTHING_WINDOW = 10          # Temporal smoothing
SMOOTH_FIT_WINDOW = 15         # User definable smoothing window for fit curves
POLY_DEGREE = 5                # User defined polynomial degree for fit lines
SPATIAL_SIGMA = 0.5           # Spatial smoothing
ROI_SIZE = 3                  # Region of Interest size
MIN_TEMP_THRESHOLD = 25.0     # Minimum temp for ROI detection
DOWN_SAMPLE_PERIOD = 90.0     # Plotting interval in seconds (multiple of SAMPLING_RATE to reduce jitter)

# Plotting
COLOR_MAP = 'inferno'
HEAT_VMIN = 25.0
HEAT_VMAX = 45.0
RAW_HEAT_VMIN = 30.0  # Raw values are higher
RAW_HEAT_VMAX = 60.0  # Raw values are higher
# =============================================================================

class CompleteThermalAnalyzer:
    """Complete analyzer focusing on Max Pixel and ROI temperatures."""
    
    @staticmethod
    def apply_emissivity_correction(t_observed, epsilon_sensor=1.0, epsilon_true=0.95):
        """
        Apply Stefan-Boltzmann correction for mouse emissivity.
        
        Your Arduino uses MLX90621 which assumes ε=1.0.
        Mouse has ε≈0.95, so temperatures are UNDERESTIMATED.
        
        Correction: T_true = T_observed × (1.0/ε_mouse)^(1/4)
        
        For ε=0.95: T_true ≈ T_observed × 1.0129 (+1.29%)
        """
        if epsilon_true >= 0.99 or abs(epsilon_sensor - epsilon_true) < 0.01:
            return t_observed
        
        # Convert to Kelvin
        t_kelvin = t_observed + 273.15
        
        # Apply Stefan-Boltzmann correction
        ratio = epsilon_sensor / epsilon_true
        t_corrected_kelvin = (t_kelvin**4 * ratio)**0.25
        
        # Convert back to Celsius
        t_corrected = t_corrected_kelvin - 273.15
        
        return t_corrected
    
    @staticmethod
    def calculate_expected_correction(temperature, epsilon_true=0.95):
        """Calculate expected correction for a given temperature."""
        if epsilon_true >= 0.99:
            return 0.0
        
        # Convert to Kelvin
        t_k = temperature + 273.15
        
        # Correction factor
        factor = (1.0 / epsilon_true)**0.25
        
        # Calculate corrected temperature
        t_corrected_k = t_k * factor
        t_corrected = t_corrected_k - 273.15
        
        return t_corrected - temperature
    
    @staticmethod
    def find_max_pixel(grid):
        """Find the brightest (hottest) pixel coordinates and temperature."""
        max_idx = np.argmax(grid)
        max_row = max_idx // grid.shape[1]
        max_col = max_idx % grid.shape[1]
        max_temp = grid[max_row, max_col]
        
        return (max_row, max_col), max_temp
    
    @staticmethod
    def find_roi(grid, min_temp=30.0, roi_size=3):
        """Find hottest region for ROI analysis - returns ROI MEAN and ROI MAX."""
        # Find hottest pixel first
        max_coords, max_temp = CompleteThermalAnalyzer.find_max_pixel(grid)
        max_row, max_col = max_coords
        
        rows, cols = grid.shape
        
        # Create ROI mask
        roi_mask = np.zeros_like(grid, dtype=bool)
        r_min = max(0, max_row - roi_size//2)
        r_max = min(rows, max_row + roi_size//2 + 1)
        c_min = max(0, max_col - roi_size//2)
        c_max = min(cols, max_col + roi_size//2 + 1)
        roi_mask[r_min:r_max, c_min:c_max] = True
        
        # Calculate ROI statistics
        roi_temps = grid[roi_mask]
        roi_mean = np.mean(roi_temps)
        roi_max = np.max(roi_temps)
        
        return (max_row, max_col), roi_mean, roi_max, roi_mask
    
    @staticmethod
    def temporal_smoothing(data, window=5):
        """Apply Savitzky-Golay smoothing."""
        if len(data) < window:
            return data
        
        if window % 2 == 0:
            window += 1
        
        return signal.savgol_filter(data, window, 2)
    
    @staticmethod
    def spatial_smoothing(grid, sigma=0.5):
        """Apply Gaussian spatial smoothing."""
        return ndimage.gaussian_filter(grid, sigma=sigma, mode='reflect')
    
    @staticmethod
    def parse_arduino_data(content):
        """Parse Arduino output format - SEPARATE raw and adjusted processing."""
        # Split by recordings
        if "=== NEW RECORDING ===" in content:
            recordings = re.split(r'===\s*NEW RECORDING\s*===', content)
        else:
            recordings = [content]
        
        recordings = [r.strip() for r in recordings if r.strip()]
        
        data_list = []
        
        for i, rec in enumerate(recordings):
            if not rec.strip():
                continue
            
            entry = {
                'frame_id': i,
                'time_s': i * SAMPLING_RATE,
                'recording_index': i,
            }
            
            # Extract IR temperatures
            ambient_match = re.search(r'IR Ambient:\s*([-\d.]+)', rec)
            object_match = re.search(r'IR Object:\s*([-\d.]+)', rec)
            
            if ambient_match:
                entry['ir_ambient'] = float(ambient_match.group(1))
            else:
                entry['ir_ambient'] = np.nan
            
            if object_match:
                entry['ir_object'] = float(object_match.group(1))
            else:
                entry['ir_object'] = np.nan
            
            # Extract Adjusted Pixel Grid - THIS IS ARDUINO'S CALIBRATED OUTPUT
            adj_grid = CompleteThermalAnalyzer.extract_grid(rec, "Adjusted Pixel Grid")
            if adj_grid is not None:
                entry['adjusted_grid'] = adj_grid
                # FOCUS ON MAX VALUES, NOT MEANS
                max_coords, max_temp = CompleteThermalAnalyzer.find_max_pixel(adj_grid)
                entry['adjusted_max'] = max_temp
                entry['adjusted_max_pos'] = max_coords
                entry['adjusted_mean'] = np.mean(adj_grid)  # Keep for reference only
            
            # Extract Raw Pixel Grid - THIS IS DIRECT FROM SENSOR
            raw_grid = CompleteThermalAnalyzer.extract_grid(rec, "Raw Pixel Grid")
            if raw_grid is not None:
                entry['raw_grid'] = raw_grid
                max_coords, max_temp = CompleteThermalAnalyzer.find_max_pixel(raw_grid)
                entry['raw_max'] = max_temp
                entry['raw_max_pos'] = max_coords
                entry['raw_mean'] = np.mean(raw_grid)  # Keep for reference only
            
            # Extract median and max temperatures from text
            median_match = re.search(r'Robust Median Temp \(calibrated\):\s*([-\d.]+)', rec)
            max_match = re.search(r'Max Temp \(calibrated\):\s*([-\d.]+).*row (\d+).*col (\d+)', rec, re.DOTALL)
            mean_match = re.search(r'Mean Temp \(calibrated\):\s*([-\d.]+)', rec)
            
            if median_match:
                entry['reported_median'] = float(median_match.group(1))
            if max_match:
                entry['reported_max'] = float(max_match.group(1))
                entry['reported_max_pos'] = (int(max_match.group(2)), int(max_match.group(3)))
            if mean_match:
                entry['reported_mean'] = float(mean_match.group(1))
            
            # We need BOTH grids for proper analysis
            if 'adjusted_grid' in entry and 'raw_grid' in entry:
                data_list.append(entry)
        
        return data_list
    
    @staticmethod
    def extract_grid(text, grid_name):
        """Extract 4x16 grid from text."""
        lines = text.split('\n')
        grid_start = -1
        
        # Find grid start
        for i, line in enumerate(lines):
            if grid_name in line and "4 rows" in line:
                grid_start = i + 1
                break
        
        if grid_start == -1 or grid_start >= len(lines):
            return None
        
        # Extract 4 lines of grid data
        grid_data = []
        for i in range(4):
            if grid_start + i < len(lines):
                line = lines[grid_start + i].strip()
                # Extract numbers
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                if len(numbers) >= 16:
                    grid_data.append([float(x) for x in numbers[:16]])
        
        if len(grid_data) == 4:
            return np.array(grid_data)
        
        return None

def analyze_with_focus_on_max_and_roi(filepath, regions=None):
    """
    Main analysis function focusing on MAX PIXEL and ROI temperatures.
    Returns separate DataFrames for Raw and Adjusted grids.
    """
    if regions is None:
        regions = [('full', slice(0,4), slice(0,16))]
    
    print(f"\n{'='*80}")
    print("FOCUSED THERMAL ANALYSIS: MAX PIXEL & ROI TEMPERATURES")
    print(f"{'='*80}")
    print(f"File: {os.path.basename(filepath)}")
    print(f"Mouse emissivity (ε): {MOUSE_EMISSIVITY}")
    print(f"Focus: Max Pixel and ROI (3x3) temperatures")
    print(f"Regions: {[r[0] for r in regions]}")
    print(f"{'='*80}")
    
    # Read file
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Parse data
    analyzer = CompleteThermalAnalyzer()
    data_list = analyzer.parse_arduino_data(content)
    
    if not data_list:
        print("ERROR: No valid data found with both Raw and Adjusted grids.")
        return None
    
    print(f"\n[1] Parsed {len(data_list)} recordings")
    
    # Process for each region
    processed = {}
    for reg_name, r_slice, c_slice in regions:
        # SEPARATE processing for Raw and Adjusted grids
        raw_processed_data = []
        adj_processed_data = []
        
        for i, entry in enumerate(data_list):
            # =====================================================================
            # PROCESS RAW PIXEL GRID (direct from sensor)
            # =====================================================================
            if 'raw_grid' in entry:
                raw_entry = entry.copy()
                raw_entry['grid_type'] = 'raw'
                
                raw_grid = entry['raw_grid'][r_slice, c_slice]
                if raw_grid.size > 0:
                    raw_entry['original_grid'] = raw_grid.copy()
                    max_coords, max_temp = analyzer.find_max_pixel(raw_grid)
                    raw_entry['original_max'] = max_temp
                    raw_entry['original_max_pos'] = max_coords
                    
                    # Apply emissivity correction to raw values
                    if APPLY_EMISSIVITY_CORRECTION:
                        raw_corrected = analyzer.apply_emissivity_correction(
                            raw_grid,
                            epsilon_sensor=SENSOR_EMISSIVITY,
                            epsilon_true=MOUSE_EMISSIVITY
                        )
                        raw_entry['corrected_grid'] = raw_corrected
                        max_coords_corr, max_temp_corr = analyzer.find_max_pixel(raw_corrected)
                        raw_entry['corrected_max'] = max_temp_corr
                        raw_entry['corrected_max_pos'] = max_coords_corr
                        raw_entry['correction_applied'] = True
                    else:
                        raw_entry['corrected_grid'] = raw_grid.copy()
                        raw_entry['corrected_max'] = raw_entry['original_max']
                        raw_entry['correction_applied'] = False
                    
                    # Spatial smoothing
                    if SPATIAL_SIGMA > 0:
                        raw_smoothed = analyzer.spatial_smoothing(
                            raw_entry['corrected_grid'], SPATIAL_SIGMA
                        )
                        raw_entry['smoothed_grid'] = raw_smoothed
                    else:
                        raw_smoothed = raw_entry['corrected_grid']
                    
                    # Find ROI in raw data (using corrected grid)
                    roi_center, roi_mean, roi_max, roi_mask = analyzer.find_roi(
                        raw_smoothed,
                        min_temp=MIN_TEMP_THRESHOLD,
                        roi_size=ROI_SIZE
                    )
                    
                    raw_entry['roi_center'] = roi_center
                    raw_entry['roi_mean'] = roi_mean
                    raw_entry['roi_max'] = roi_max
                    raw_entry['roi_mask'] = roi_mask
                    
                    raw_processed_data.append(raw_entry)
            
            # =====================================================================
            # PROCESS ADJUSTED PIXEL GRID (Arduino's calibrated output)
            # =====================================================================
            if 'adjusted_grid' in entry:
                adj_entry = entry.copy()
                adj_entry['grid_type'] = 'adjusted'
                
                adj_grid = entry['adjusted_grid'][r_slice, c_slice]
                if adj_grid.size > 0:
                    adj_entry['original_grid'] = adj_grid.copy()
                    max_coords, max_temp = analyzer.find_max_pixel(adj_grid)
                    adj_entry['original_max'] = max_temp
                    adj_entry['original_max_pos'] = max_coords
                    
                    # Apply emissivity correction to Arduino's already-calibrated values
                    if APPLY_EMISSIVITY_CORRECTION:
                        adj_corrected = analyzer.apply_emissivity_correction(
                            adj_grid,
                            epsilon_sensor=SENSOR_EMISSIVITY,
                            epsilon_true=MOUSE_EMISSIVITY
                        )
                        adj_entry['corrected_grid'] = adj_corrected
                        max_coords_corr, max_temp_corr = analyzer.find_max_pixel(adj_corrected)
                        adj_entry['corrected_max'] = max_temp_corr
                        adj_entry['corrected_max_pos'] = max_coords_corr
                        adj_entry['correction_applied'] = True
                        adj_entry['max_correction'] = max_temp_corr - max_temp
                    else:
                        adj_entry['corrected_grid'] = adj_grid.copy()
                        adj_entry['corrected_max'] = adj_entry['original_max']
                        adj_entry['correction_applied'] = False
                        adj_entry['max_correction'] = 0.0
                    
                    # Spatial smoothing
                    if SPATIAL_SIGMA > 0:
                        adj_smoothed = analyzer.spatial_smoothing(
                            adj_entry['corrected_grid'], SPATIAL_SIGMA
                        )
                        adj_entry['smoothed_grid'] = adj_smoothed
                    else:
                        adj_smoothed = adj_entry['corrected_grid']
                    
                    # Find ROI in adjusted data
                    roi_center, roi_mean, roi_max, roi_mask = analyzer.find_roi(
                        adj_smoothed,
                        min_temp=MIN_TEMP_THRESHOLD,
                        roi_size=ROI_SIZE
                    )
                    
                    adj_entry['roi_center'] = roi_center
                    adj_entry['roi_mean'] = roi_mean
                    adj_entry['roi_max'] = roi_max
                    adj_entry['roi_mask'] = roi_mask
                    
                    adj_processed_data.append(adj_entry)
            
            if i % 10 == 0 or i == len(data_list) - 1:
                print(f"  Frame {i:3d} ({reg_name}): "
                      f"Raw Max={entry.get('raw_max', np.nan):.1f}°C → "
                      f"Adj Max={entry.get('adjusted_max', np.nan):.1f}°C → "
                      f"Corr Max={adj_entry.get('corrected_max', np.nan):.1f}°C" if 'adj_entry' in locals() else "")
        
        # Apply temporal smoothing to MAX and ROI series separately
        if len(raw_processed_data) > SMOOTHING_WINDOW:
            raw_max_series = [d['corrected_max'] for d in raw_processed_data]
            raw_roi_series = [d['roi_max'] for d in raw_processed_data]
            
            raw_max_smoothed = analyzer.temporal_smoothing(raw_max_series, SMOOTHING_WINDOW)
            raw_roi_smoothed = analyzer.temporal_smoothing(raw_roi_series, SMOOTHING_WINDOW)
            
            for i, entry in enumerate(raw_processed_data):
                entry['max_smoothed'] = raw_max_smoothed[i]
                entry['roi_max_smoothed'] = raw_roi_smoothed[i]
        
        if len(adj_processed_data) > SMOOTHING_WINDOW:
            adj_max_series = [d['corrected_max'] for d in adj_processed_data]
            adj_roi_series = [d['roi_max'] for d in adj_processed_data]
            
            adj_max_smoothed = analyzer.temporal_smoothing(adj_max_series, SMOOTHING_WINDOW)
            adj_roi_smoothed = analyzer.temporal_smoothing(adj_roi_series, SMOOTHING_WINDOW)
            
            for i, entry in enumerate(adj_processed_data):
                entry['max_smoothed'] = adj_max_smoothed[i]
                entry['roi_max_smoothed'] = adj_roi_smoothed[i]
        
        processed[reg_name] = (pd.DataFrame(raw_processed_data), pd.DataFrame(adj_processed_data))
    
    if len(regions) == 1:
        return processed['full']
    return processed

def create_max_and_roi_plots(raw_df, adj_df, filepath, output_dir):
    """
    Create FOCUSED plots for Max Pixel and ROI temperatures.
    Includes 3D visualizations and time series.
    """
    print(f"\n[2] Creating FOCUSED plots for Max Pixel and ROI...")
    
    # Create plot directories
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    max_roi_dir = os.path.join(plot_dir, 'max_roi_focused')
    three_d_dir = os.path.join(plot_dir, '3d_visualizations')
    comparison_dir = os.path.join(plot_dir, 'comparison_analysis')
    
    for dir_path in [max_roi_dir, three_d_dir, comparison_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # =========================================================================
    # 1. FOCUSED TIME SERIES: MAX PIXEL vs ROI
    # =========================================================================
    print(f"  Creating Max Pixel vs ROI time series...")
    
    if not adj_df.empty:
        times = adj_df['time_s'].values
        chunk_size = max(1, round(DOWN_SAMPLE_PERIOD / SAMPLING_RATE))
        downsampled_times = []
        downsampled_original_max_med = []
        downsampled_original_max_std = []
        downsampled_corrected_max_med = []
        downsampled_corrected_max_std = []
        downsampled_roi_max_med = []
        downsampled_roi_max_std = []
        downsampled_max_smoothed = [] if 'max_smoothed' in adj_df.columns else None
        downsampled_roi_max_smoothed = [] if 'roi_max_smoothed' in adj_df.columns else None
        downsampled_grids = []
        downsampled_max_pos = []
        downsampled_roi_center = []
        downsampled_roi_mask = []
        downsampled_corrected_max_from_avg = []
        downsampled_roi_max_from_avg = []
        analyzer = CompleteThermalAnalyzer()
        for start in range(0, len(adj_df), chunk_size):
            end = start + chunk_size
            chunk = adj_df.iloc[start:end]
            downsampled_times.append(np.median(chunk['time_s']))
            downsampled_original_max_med.append(np.median(chunk['original_max']))
            downsampled_original_max_std.append(np.std(chunk['original_max']))
            downsampled_corrected_max_med.append(np.median(chunk['corrected_max']))
            downsampled_corrected_max_std.append(np.std(chunk['corrected_max']))
            downsampled_roi_max_med.append(np.median(chunk['roi_max']))
            downsampled_roi_max_std.append(np.std(chunk['roi_max']))
            if downsampled_max_smoothed is not None:
                downsampled_max_smoothed.append(np.median(chunk['max_smoothed']))
            if downsampled_roi_max_smoothed is not None:
                downsampled_roi_max_smoothed.append(np.median(chunk['roi_max_smoothed']))
            grid_key = 'smoothed_grid' if 'smoothed_grid' in chunk.columns else 'corrected_grid'
            chunk_grids = np.stack(chunk[grid_key].values)
            avg_grid = np.median(chunk_grids, axis=0)
            downsampled_grids.append(avg_grid)
            max_coords, max_temp = analyzer.find_max_pixel(avg_grid)
            downsampled_max_pos.append(max_coords)
            downsampled_corrected_max_from_avg.append(max_temp)
            roi_center, roi_mean, roi_max, roi_mask = analyzer.find_roi(avg_grid, MIN_TEMP_THRESHOLD, ROI_SIZE)
            downsampled_roi_center.append(roi_center)
            downsampled_roi_max_from_avg.append(roi_max)
            downsampled_roi_mask.append(roi_mask)
        times_ds = np.array(downsampled_times)
        original_max_med = np.array(downsampled_original_max_med)
        original_max_std = np.array(downsampled_original_max_std)
        corrected_max_med = np.array(downsampled_corrected_max_med)
        corrected_max_std = np.array(downsampled_corrected_max_std)
        roi_max_med = np.array(downsampled_roi_max_med)
        roi_max_std = np.array(downsampled_roi_max_std)
        if downsampled_max_smoothed is not None:
            max_smoothed_ds = np.array(downsampled_max_smoothed)
        if downsampled_roi_max_smoothed is not None:
            roi_max_smoothed_ds = np.array(downsampled_roi_max_smoothed)
        
        # Compute normalized series
        normalized_corrected_max_med = corrected_max_med - corrected_max_med[0]
        normalized_roi_max_med = roi_max_med - roi_max_med[0]
        if downsampled_max_smoothed is not None:
            normalized_max_smoothed_ds = max_smoothed_ds - max_smoothed_ds[0]
        if downsampled_roi_max_smoothed is not None:
            normalized_roi_max_smoothed_ds = roi_max_smoothed_ds - roi_max_smoothed_ds[0]
        normalized_original_max_med = original_max_med - original_max_med[0]
        
        # 1A: Main Max Pixel vs ROI Comparison
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Plot 1: Max Pixel Temperature
        ax1 = axes[0]
        
        ax1.plot(times_ds, normalized_corrected_max_med, 'r-', linewidth=3,
                label=f'Normalized ε-Corrected Max Pixel ({np.median(normalized_corrected_max_med):.1f}°C rel.)')
        
        if downsampled_max_smoothed is not None:
            ax1.plot(times_ds, normalized_max_smoothed_ds, 'r--', linewidth=2,
                    label=f'Normalized Smoothed Max ({np.median(normalized_max_smoothed_ds):.1f}°C rel.)', alpha=0.8)
        
        # Polynomial fit
        if len(times_ds) > POLY_DEGREE:
            p_max = np.polyfit(times_ds, normalized_corrected_max_med, POLY_DEGREE)
            fit_max = np.polyval(p_max, times_ds)
            ax1.plot(times_ds, fit_max, 'r:', linewidth=2, label=f'Poly fit (deg={POLY_DEGREE})')
        
        ax1.fill_between(times_ds, normalized_corrected_max_med - corrected_max_std, normalized_corrected_max_med + corrected_max_std,
                         alpha=0.1, color='red')
        
        ax1.set_ylabel('Normalized Max Pixel Temp (°C)', fontsize=12)
        ax1.set_title('MAX PIXEL TEMPERATURE: Brightest (Hottest) Pixel', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: ROI Max Temperature
        ax2 = axes[1]
        
        ax2.plot(times_ds, normalized_roi_max_med, 'g-', linewidth=3,
                label=f'Normalized ROI Max (3x3 region) ({np.median(normalized_roi_max_med):.1f}°C rel.)')
        
        if downsampled_roi_max_smoothed is not None:
            ax2.plot(times_ds, normalized_roi_max_smoothed_ds, 'g--', linewidth=2,
                    label=f'Normalized Smoothed ROI Max ({np.median(normalized_roi_max_smoothed_ds):.1f}°C rel.)', alpha=0.8)
        
        # Polynomial fit
        if len(times_ds) > POLY_DEGREE:
            p_roi = np.polyfit(times_ds, normalized_roi_max_med, POLY_DEGREE)
            fit_roi = np.polyval(p_roi, times_ds)
            ax2.plot(times_ds, fit_roi, 'g:', linewidth=2, label=f'Poly fit (deg={POLY_DEGREE})')
        
        ax2.fill_between(times_ds, normalized_roi_max_med - roi_max_std, normalized_roi_max_med + roi_max_std,
                         alpha=0.1, color='green')
        
        ax2.set_ylabel('Normalized ROI Max Temp (°C)', fontsize=12)
        ax2.set_title('ROI (3x3 REGION) MAXIMUM TEMPERATURE', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Comparison between Max Pixel and ROI Max
        ax3 = axes[2]
        
        ax3.plot(times_ds, normalized_corrected_max_med, 'r-', linewidth=2.5,
                label=f'Normalized Max Pixel ({np.median(normalized_corrected_max_med):.1f}°C rel.)')
        
        ax3.plot(times_ds, normalized_roi_max_med, 'g-', linewidth=2.5,
                label=f'Normalized ROI Max ({np.median(normalized_roi_max_med):.1f}°C rel.)')
        
        # Plot the difference
        diff = normalized_corrected_max_med - normalized_roi_max_med
        ax3.plot(times_ds, diff, 'purple', linewidth=1.5, alpha=0.7,
                label=f'Difference (Max - ROI Max)', linestyle=':')
        
        ax3.fill_between(times_ds, normalized_roi_max_med, normalized_corrected_max_med,
                        where=normalized_corrected_max_med > normalized_roi_max_med,
                        alpha=0.2, color='red', label='Max > ROI Max')
        
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.axhline(y=diff.mean(), color='purple', linestyle='-', alpha=0.5,
                   label=f'Mean diff: {diff.mean():.2f}°C')
        
        ax3.set_xlabel('Time (seconds)', fontsize=12)
        ax3.set_ylabel('Normalized Temperature (°C)', fontsize=12)
        ax3.set_title('MAX PIXEL vs ROI MAX COMPARISON', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'FOCUSED TEMPERATURE ANALYSIS: MAX PIXEL & ROI\n'
                    f'File: {os.path.basename(filepath)}, ε={MOUSE_EMISSIVITY}',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        max_roi_ts_path = os.path.join(max_roi_dir, 'max_pixel_vs_roi_time_series.png')
        plt.savefig(max_roi_ts_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        # 1B: Individual detailed plots for Max Pixel
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        # Max Pixel time series with correction
        ax1 = axes[0]
        ax1.plot(times_ds, normalized_corrected_max_med, 'r-', linewidth=3,
                label=f'Normalized ε-Corrected ({np.median(normalized_corrected_max_med):.1f}°C rel.)')
        
        ax1.fill_between(times_ds, normalized_corrected_max_med - corrected_max_std, normalized_corrected_max_med + corrected_max_std,
                         alpha=0.1, color='red')
        
        # Polynomial fit
        if len(times_ds) > POLY_DEGREE:
            p_max = np.polyfit(times_ds, normalized_corrected_max_med, POLY_DEGREE)
            fit_max = np.polyval(p_max, times_ds)
            ax1.plot(times_ds, fit_max, 'r--', linewidth=2, label=f'Poly fit (deg={POLY_DEGREE})')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Normalized Temperature (°C)')
        ax1.set_title('MAX PIXEL: ε-Corrected')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Max Pixel correction magnitude
        ax2 = axes[1]
        max_correction = corrected_max_med - original_max_med
        
        ax2.plot(times_ds, max_correction, 'purple', linewidth=2.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axhline(y=max_correction.mean(), color='r', linestyle=':',
                   linewidth=2, label=f'Mean: {max_correction.mean():.3f}°C')
        
        expected = CompleteThermalAnalyzer.calculate_expected_correction(
            np.median(original_max_med), MOUSE_EMISSIVITY
        )
        ax2.axhline(y=expected, color='g', linestyle='--',
                   linewidth=2, label=f'Expected: {expected:.3f}°C')
        
        ax2.fill_between(times_ds, 0, max_correction, where=max_correction>0,
                        alpha=0.3, color='red', label='Positive')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Correction (°C)')
        ax2.set_title(f'Max Pixel Correction (ε={MOUSE_EMISSIVITY})')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # ROI Max time series
        ax3 = axes[2]
        ax3.plot(times_ds, normalized_roi_max_med, 'g-', linewidth=3,
                label=f'Normalized ROI Max ({np.median(normalized_roi_max_med):.1f}°C rel.)')
        
        ax3.fill_between(times_ds, normalized_roi_max_med - roi_max_std, normalized_roi_max_med + roi_max_std,
                         alpha=0.1, color='green')
        
        
        
        # Polynomial fit
        if len(times_ds) > POLY_DEGREE:
            p_roi = np.polyfit(times_ds, normalized_roi_max_med, POLY_DEGREE)
            fit_roi = np.polyval(p_roi, times_ds)
            ax3.plot(times_ds, fit_roi, 'g', linestyle='--', linewidth=2, label=f'Poly fit (deg={POLY_DEGREE})')
        
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Normalized Temperature (°C)')
        ax3.set_title('ROI MAXIMUM (3x3 region)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Rate of change for Max Pixel
        ax4 = axes[3]
        if downsampled_max_smoothed is not None:
            smoothed_max = normalized_max_smoothed_ds
        else:
            smoothed_max = normalized_corrected_max_med
        
        rate_max = np.gradient(smoothed_max, times_ds) * 60  # °C per minute
        
        ax4.plot(times_ds, rate_max, 'black', linewidth=2, label='Rate')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        ax4.fill_between(times_ds, 0, rate_max, where=rate_max>0,
                        alpha=0.3, color='orange', label='Warming')
        ax4.fill_between(times_ds, 0, rate_max, where=rate_max<0,
                        alpha=0.3, color='cyan', label='Cooling')
        
        # Smoothed rate
        if len(rate_max) > SMOOTH_FIT_WINDOW:
            rate_smoothed = signal.savgol_filter(rate_max, SMOOTH_FIT_WINDOW, 2)
            ax4.plot(times_ds, rate_smoothed, 'black', linestyle='--', linewidth=2, label='Smoothed Rate')
        
        
        
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Rate of Change (°C/minute)')
        ax4.set_title('Max Pixel Rate of Change')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('DETAILED MAX PIXEL & ROI ANALYSIS', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        detail_path = os.path.join(max_roi_dir, 'detailed_max_roi_analysis.png')
        plt.savefig(detail_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    # =========================================================================
    # 2. 3D VISUALIZATIONS
    # =========================================================================
    print(f"  Creating 3D visualizations...")
    
    if not adj_df.empty and len(adj_df) > 5:
        # Select key frames for 3D visualization
        num_frames_to_plot = 10
        total_frames = len(downsampled_grids)
        key_frames = [int(i * (total_frames - 1) / (num_frames_to_plot - 1)) for i in range(num_frames_to_plot)]
        
        for frame_idx in key_frames:
            if frame_idx >= len(downsampled_grids):
                continue
            
            grid = downsampled_grids[frame_idx]
            rows, cols = grid.shape
            
            # Create 3D surface plot
            fig = plt.figure(figsize=(16, 12))
            
            # Plot 1: 3D Surface
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            
            X, Y = np.meshgrid(range(cols), range(rows))
            surf = ax1.plot_surface(X, Y, grid, cmap='inferno',
                                  linewidth=0, antialiased=True,
                                  alpha=0.8)
            
            # Mark Max Pixel
            max_row, max_col = downsampled_max_pos[frame_idx]
            ax1.scatter(max_col, max_row, grid[max_row, max_col],
                      s=200, c='red', edgecolor='white', linewidth=2,
                      label=f'Max Pixel: {grid[max_row, max_col]:.1f}°C')
            
            # Mark ROI
            roi_mask = downsampled_roi_mask[frame_idx]
            roi_coords = np.where(roi_mask)
            if len(roi_coords[0]) > 0:
                ax1.scatter(roi_coords[1], roi_coords[0], 
                          grid[roi_coords[0], roi_coords[1]],
                          s=100, c='yellow', edgecolor='black', alpha=0.6,
                          linewidth=1, label='ROI (3x3)')
            
            ax1.set_xlabel('Column', fontsize=10)
            ax1.set_ylabel('Row', fontsize=10)
            ax1.set_zlabel('Temperature (°C)', fontsize=10)
            ax1.set_title(f'3D Thermal Surface - Frame {frame_idx}\n'
                         f'Time: {downsampled_times[frame_idx]:.1f}s, Max: {downsampled_corrected_max_from_avg[frame_idx]:.1f}°C',
                         fontsize=12)
            ax1.legend(loc='upper right', fontsize=9)
            
            plt.colorbar(surf, ax=ax1, shrink=0.6, label='°C')
            
            # Plot 2: 3D Wireframe
            ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            
            wire = ax2.plot_wireframe(X, Y, grid, color='blue', alpha=0.7,
                                    linewidth=1, antialiased=True)
            
            # Add surface with transparency
            surf2 = ax2.plot_surface(X, Y, grid, cmap='viridis',
                                   alpha=0.4, linewidth=0)
            
            max_row, max_col = downsampled_max_pos[frame_idx]
            ax2.scatter(max_col, max_row, grid[max_row, max_col],
                      s=300, c='red', edgecolor='white', linewidth=3,
                      marker='*', label='Max Pixel')
            
            ax2.set_xlabel('Column', fontsize=10)
            ax2.set_ylabel('Row', fontsize=10)
            ax2.set_zlabel('Temperature (°C)', fontsize=10)
            ax2.set_title('3D Wireframe + Surface', fontsize=12)
            ax2.legend(loc='upper right', fontsize=9)
            
            plt.colorbar(surf2, ax=ax2, shrink=0.6, label='°C')
            
            # Plot 3: 2D Heatmap with 3D perspective
            ax3 = fig.add_subplot(2, 2, 3)
            
            im = ax3.imshow(grid, cmap='inferno', aspect='equal',
                          vmin=HEAT_VMIN, vmax=HEAT_VMAX)
            
            # Add Max Pixel marker
            max_row, max_col = downsampled_max_pos[frame_idx]
            ax3.scatter(max_col, max_row, s=400, facecolors='none',
                      edgecolors='yellow', linewidth=3, marker='s',
                      label=f'Max: {grid[max_row, max_col]:.1f}°C')
            
            # Add ROI rectangle
            r, c = downsampled_roi_center[frame_idx]
            rect = plt.Rectangle((c-1.5, r-1.5), 3, 3,
                               linewidth=2, edgecolor='white',
                               facecolor='none', linestyle='--',
                               label='ROI (3x3)')
            ax3.add_patch(rect)
            
            # Add temperature values for Max Pixel and surrounding area
            for r in range(max(0, max_row-1), min(rows, max_row+2)):
                for c in range(max(0, max_col-1), min(cols, max_col+2)):
                    val = grid[r, c]
                    color = "white" if val < (HEAT_VMIN + HEAT_VMAX)/2 else "black"
                    fontweight = 'bold' if (r == max_row and c == max_col) else 'normal'
                    ax3.text(c, r, f"{val:.1f}", ha="center", va="center",
                           color=color, fontsize=8, fontweight=fontweight)
            
            ax3.set_xlabel('Column', fontsize=10)
            ax3.set_ylabel('Row', fontsize=10)
            ax3.set_title('2D Heatmap with Max Pixel & ROI', fontsize=12)
            ax3.legend(loc='upper right', fontsize=9)
            plt.colorbar(im, ax=ax3, shrink=0.8, label='°C')
            
            # Plot 4: Temperature profile along row with Max Pixel
            ax4 = fig.add_subplot(2, 2, 4)
            
            # Plot row with Max Pixel
            max_row, max_col = downsampled_max_pos[frame_idx]
            row_profile = grid[max_row, :]
            ax4.plot(range(cols), row_profile, 'b-', linewidth=2.5,
                    label=f'Row {max_row} (contains Max Pixel)')
            ax4.scatter(max_col, row_profile[max_col], s=150,
                      color='red', edgecolor='black', linewidth=2,
                      zorder=5, label=f'Max: {row_profile[max_col]:.1f}°C')
            
            # Plot all rows for comparison
            for r in range(rows):
                if r != max_row:
                    ax4.plot(range(cols), grid[r, :], 'gray', alpha=0.4,
                            linewidth=1, label=f'Row {r}' if r == 0 else "")
            
            ax4.set_xlabel('Column', fontsize=10)
            ax4.set_ylabel('Temperature (°C)', fontsize=10)
            ax4.set_title('Temperature Profile Along Rows', fontsize=12)
            ax4.legend(loc='best', fontsize=9)
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'3D THERMAL VISUALIZATION - Frame {frame_idx}\n'
                       f'Max Pixel: {downsampled_corrected_max_from_avg[frame_idx]:.1f}°C, ROI Max: {downsampled_roi_max_from_avg[frame_idx]:.1f}°C',
                       fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            three_d_path = os.path.join(three_d_dir, f'3d_visualization_frame_{frame_idx:04d}.png')
            plt.savefig(three_d_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"    Created {len(key_frames)} 3D visualizations")
    
    # =========================================================================
    # 3. MAX PIXEL MOVEMENT TRACKING
    # =========================================================================
    print(f"  Creating Max Pixel movement tracking...")
    
    if not adj_df.empty and 'corrected_max_pos' in adj_df.columns:
        try:
            # Extract Max Pixel positions
            rows_ds = [p[0] for p in downsampled_max_pos]
            cols_ds = [p[1] for p in downsampled_max_pos]
            rows_ds = np.array(rows_ds)
            cols_ds = np.array(cols_ds)
            
            # Create boolean mask for valid positions
            valid_mask = ~(np.isnan(rows_ds) | np.isnan(cols_ds))
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 1:
                grid = downsampled_grids[0]  # For shape
                rows, cols = grid.shape
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Plot 1: Movement trajectory
                ax1 = axes[0, 0]
                temps = corrected_max_med
                normalized_temps = temps - temps[0]
                valid_rows = rows_ds[valid_indices]
                valid_cols = cols_ds[valid_indices]
                
                scatter1 = ax1.scatter(valid_cols, valid_rows,
                                      c=normalized_temps, cmap='hot', s=100, alpha=0.8,
                                      edgecolors='k', linewidth=1)
                
                # Connect points in time order
                for i in range(len(valid_indices)-1):
                    idx1, idx2 = valid_indices[i], valid_indices[i+1]
                    ax1.plot([cols_ds[idx1], cols_ds[idx2]], [rows_ds[idx1], rows_ds[idx2]],
                            'k-', alpha=0.3, linewidth=1)
                
                # Mark start and end
                if len(valid_indices) > 0:
                    ax1.scatter(cols_ds[valid_indices[0]], rows_ds[valid_indices[0]], 
                              s=200, facecolor='green', edgecolor='k', marker='o', 
                              label='Start', zorder=5)
                    ax1.scatter(cols_ds[valid_indices[-1]], rows_ds[valid_indices[-1]], 
                              s=200, facecolor='red', edgecolor='k', marker='s', 
                              label='End', zorder=5)
                
                ax1.set_xlabel('Column', fontsize=12)
                ax1.set_ylabel('Row', fontsize=12)
                ax1.set_title('Max Pixel Movement (Hotspot Tracking)', fontsize=14, fontweight='bold')
                ax1.set_xlim(-0.5, cols - 0.5)
                ax1.set_ylim(-0.5, rows - 0.5)
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc='best')
                
                cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
                cbar1.set_label('Normalized Max Pixel Temperature (°C)', fontsize=10)
                
                # Plot 2: Movement heatmap
                ax2 = axes[0, 1]
                movement_heatmap = np.zeros((rows, cols))
                
                for r, c in zip(valid_rows, valid_cols):
                    if 0 <= r < rows and 0 <= c < cols:
                        movement_heatmap[int(r), int(c)] += 1
                
                im2 = ax2.imshow(movement_heatmap, cmap='YlOrRd', aspect='equal', alpha=0.8)
                
                # Add count values
                for r in range(rows):
                    for c in range(cols):
                        count = movement_heatmap[r, c]
                        if count > 0:
                            ax2.text(c, r, f"{int(count)}", ha="center", va="center",
                                   color="white" if count > np.max(movement_heatmap)/2 else "black",
                                   fontsize=9, fontweight='bold')
                
                ax2.set_xlabel('Column', fontsize=12)
                ax2.set_ylabel('Row', fontsize=12)
                ax2.set_title('Max Pixel Frequency Heatmap', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)
                
                cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
                cbar2.set_label('Visit Count', fontsize=10)
                
                # Plot 3: Temperature vs Position
                ax3 = axes[1, 0]
                scatter3 = ax3.scatter(valid_cols, normalized_temps,
                                      c=times_ds, 
                                      cmap='viridis', s=80, alpha=0.8,
                                      edgecolors='k', linewidth=0.5)
                
                # Add regression line
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        valid_cols, normalized_temps
                    )
                    x_fit = np.array([min(valid_cols), max(valid_cols)])
                    y_fit = slope * x_fit + intercept
                    ax3.plot(x_fit, y_fit, 'r-', alpha=0.8,
                            label=f'Fit: slope={slope:.3f}, R²={r_value**2:.3f}')
                except:
                    pass
                
                ax3.set_xlabel('Column Position', fontsize=12)
                ax3.set_ylabel('Normalized Max Pixel Temperature (°C)', fontsize=12)
                ax3.set_title('Temperature vs Column Position', fontsize=14, fontweight='bold')
                ax3.legend(loc='best')
                ax3.grid(True, alpha=0.3)
                
                cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
                cbar3.set_label('Time (s)', fontsize=10)
                
                # Plot 4: Position stability
                ax4 = axes[1, 1]
                
                # Calculate movement distance between frames
                distances = []
                for i in range(1, len(downsampled_max_pos)):
                    prev = downsampled_max_pos[i-1]
                    curr = downsampled_max_pos[i]
                    dist = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                    distances.append(dist)
                
                ax4.plot(times_ds[1:], distances, 'b-', linewidth=2, label='Movement Distance')
                ax4.fill_between(times_ds[1:], 0, distances, alpha=0.3, color='blue')
                
                mean_dist = np.mean(distances) if distances else 0
                ax4.axhline(y=mean_dist, color='r', linestyle='--',
                           label=f'Mean: {mean_dist:.2f} pixels/frame')
                ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                ax4.set_xlabel('Time (seconds)', fontsize=12)
                ax4.set_ylabel('Movement Distance (pixels)', fontsize=12)
                ax4.set_title('Max Pixel Movement Stability', fontsize=14, fontweight='bold')
                ax4.legend(loc='best')
                ax4.grid(True, alpha=0.3)
                
                plt.suptitle('MAX PIXEL MOVEMENT ANALYSIS & TRACKING', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                movement_path = os.path.join(max_roi_dir, 'max_pixel_movement_tracking.png')
                plt.savefig(movement_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"    Created Max Pixel movement tracking plot")
        except Exception as e:
            print(f"    Warning: Could not create movement tracking: {str(e)}")
    
    # =========================================================================
    # 4. COMPARISON WITH RAW DATA
    # =========================================================================
    print(f"  Creating Raw vs Adjusted comparison...")
    
    if not raw_df.empty and not adj_df.empty:
        downsampled_raw_corrected_max_med = []
        downsampled_raw_corrected_max_std = []
        downsampled_raw_roi_max_med = []
        downsampled_raw_roi_max_std = []
        for start in range(0, len(raw_df), chunk_size):
            end = start + chunk_size
            chunk = raw_df.iloc[start:end]
            downsampled_raw_corrected_max_med.append(np.median(chunk['corrected_max']))
            downsampled_raw_corrected_max_std.append(np.std(chunk['corrected_max']))
            downsampled_raw_roi_max_med.append(np.median(chunk['roi_max']))
            downsampled_raw_roi_max_std.append(np.std(chunk['roi_max']))
        downsampled_raw_corrected_max_med = np.array(downsampled_raw_corrected_max_med)
        downsampled_raw_corrected_max_std = np.array(downsampled_raw_corrected_max_std)
        downsampled_raw_roi_max_med = np.array(downsampled_raw_roi_max_med)
        downsampled_raw_roi_max_std = np.array(downsampled_raw_roi_max_std)
        
        # Compute normalized series for raw
        normalized_downsampled_raw_corrected_max_med = downsampled_raw_corrected_max_med - downsampled_raw_corrected_max_med[0]
        normalized_downsampled_raw_roi_max_med = downsampled_raw_roi_max_med - downsampled_raw_roi_max_med[0]
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        
        # Plot 1: Max Pixel comparison
        ax1 = axes[0]
        ax1.plot(times_ds, normalized_downsampled_raw_corrected_max_med, 'k-', linewidth=2,
                label=f'Normalized Raw Max ({np.median(normalized_downsampled_raw_corrected_max_med):.1f}°C rel.)', alpha=0.7)
        ax1.fill_between(times_ds, normalized_downsampled_raw_corrected_max_med - downsampled_raw_corrected_max_std, 
                         normalized_downsampled_raw_corrected_max_med + downsampled_raw_corrected_max_std, alpha=0.1, color='black')
        ax1.plot(times_ds, normalized_corrected_max_med, 'r-', linewidth=3,
                label=f'Normalized ε-Corrected Max ({np.median(normalized_corrected_max_med):.1f}°C rel.)')
        ax1.fill_between(times_ds, normalized_corrected_max_med - corrected_max_std, normalized_corrected_max_med + corrected_max_std, alpha=0.1, color='red')
        
        ax1.set_ylabel('Normalized Max Pixel Temp (°C)', fontsize=12)
        ax1.set_title('MAX PIXEL: Raw → ε-Corrected', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: ROI Max comparison
        ax2 = axes[1]
        ax2.plot(times_ds, normalized_downsampled_raw_roi_max_med, 'k-', linewidth=2,
                label=f'Normalized Raw ROI Max ({np.median(normalized_downsampled_raw_roi_max_med):.1f}°C rel.)', alpha=0.7)
        ax2.fill_between(times_ds, normalized_downsampled_raw_roi_max_med - downsampled_raw_roi_max_std, 
                         normalized_downsampled_raw_roi_max_med + downsampled_raw_roi_max_std, alpha=0.1, color='black')
        ax2.plot(times_ds, normalized_roi_max_med, 'g-', linewidth=3,
                label=f'Normalized Adjusted ROI Max ({np.median(normalized_roi_max_med):.1f}°C rel.)')
        ax2.fill_between(times_ds, normalized_roi_max_med - roi_max_std, normalized_roi_max_med + roi_max_std, alpha=0.1, color='green')
        
        ax2.set_ylabel('Normalized ROI Max Temp (°C)', fontsize=12)
        ax2.set_title('ROI MAX (3x3): Raw vs Adjusted', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Correction chain
        ax3 = axes[2]
        
        # Calculate corrections
        total_correction = corrected_max_med - downsampled_raw_corrected_max_med
        
        ax3.plot(times_ds, total_correction, 'purple', linewidth=3,
                label=f'Total Correction (Mean: {np.mean(total_correction):.1f}°C)')
        
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Time (seconds)', fontsize=12)
        ax3.set_ylabel('Correction (°C)', fontsize=12)
        ax3.set_title('CORRECTION CHAIN ANALYSIS', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('COMPLETE CORRECTION ANALYSIS: MAX PIXEL FOCUS', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        comparison_path = os.path.join(comparison_dir, 'max_pixel_correction_chain.png')
        plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ All focused plots created successfully!")
    return max_roi_dir, three_d_dir, comparison_dir

def export_focused_results(raw_df, adj_df, filepath, output_dir):
    """Export focused results for Max Pixel and ROI temperatures."""
    print(f"\n[3] Exporting focused results...")
    
    # Create export directories
    export_dir = os.path.join(output_dir, 'exports')
    os.makedirs(export_dir, exist_ok=True)
    
    focused_export_dir = os.path.join(export_dir, 'max_roi_focused_data')
    os.makedirs(focused_export_dir, exist_ok=True)
    
    # =========================================================================
    # 1. ADJUSTED GRID FOCUSED DATA
    # =========================================================================
    if not adj_df.empty:
        print(f"  Exporting Adjusted Grid focused data...")
        
        focused_data = []
        for idx, row in adj_df.iterrows():
            export_row = {
                'frame': idx,
                'time_s': row['time_s'],
                'arduino_max': row['original_max'],
                'arduino_max_row': row['original_max_pos'][0] if isinstance(row['original_max_pos'], tuple) else np.nan,
                'arduino_max_col': row['original_max_pos'][1] if isinstance(row['original_max_pos'], tuple) else np.nan,
                'corrected_max': row['corrected_max'],
                'corrected_max_row': row['corrected_max_pos'][0] if isinstance(row['corrected_max_pos'], tuple) else np.nan,
                'corrected_max_col': row['corrected_max_pos'][1] if isinstance(row['corrected_max_pos'], tuple) else np.nan,
                'max_correction': row.get('max_correction', row['corrected_max'] - row['original_max']),
                'roi_max': row['roi_max'],
                'roi_mean': row['roi_mean'],
                'roi_center_row': row['roi_center'][0] if isinstance(row['roi_center'], tuple) else np.nan,
                'roi_center_col': row['roi_center'][1] if isinstance(row['roi_center'], tuple) else np.nan,
                'ambient_temp': row.get('ir_ambient', np.nan),
            }
            
            if 'max_smoothed' in row:
                export_row['max_smoothed'] = row['max_smoothed']
            if 'roi_max_smoothed' in row:
                export_row['roi_max_smoothed'] = row['roi_max_smoothed']
            
            focused_data.append(export_row)
        
        focused_df = pd.DataFrame(focused_data)
        if not focused_df.empty:
            start_corrected_max = focused_df['corrected_max'].iloc[0]
            focused_df['normalized_corrected_max'] = focused_df['corrected_max'] - start_corrected_max
            start_roi_max = focused_df['roi_max'].iloc[0]
            focused_df['normalized_roi_max'] = focused_df['roi_max'] - start_roi_max
            if 'max_smoothed' in focused_df:
                start_max_smoothed = focused_df['max_smoothed'].iloc[0]
                focused_df['normalized_max_smoothed'] = focused_df['max_smoothed'] - start_max_smoothed
            if 'roi_max_smoothed' in focused_df:
                start_roi_max_smoothed = focused_df['roi_max_smoothed'].iloc[0]
                focused_df['normalized_roi_max_smoothed'] = focused_df['roi_max_smoothed'] - start_roi_max_smoothed
        
        focused_path = os.path.join(focused_export_dir, 'max_pixel_roi_focused_data.csv')
        focused_df.to_csv(focused_path, index=False)
        print(f"    ✓ Focused data: {focused_path}")
        
        # Summary statistics
        summary = {
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source_file': os.path.basename(filepath),
            'total_frames': len(adj_df),
            'mouse_emissivity': MOUSE_EMISSIVITY,
            
            'arduino_max_mean': adj_df['original_max'].mean(),
            'arduino_max_std': adj_df['original_max'].std(),
            'arduino_max_min': adj_df['original_max'].min(),
            'arduino_max_max': adj_df['original_max'].max(),
            
            'corrected_max_mean': adj_df['corrected_max'].mean(),
            'corrected_max_std': adj_df['corrected_max'].std(),
            'corrected_max_min': adj_df['corrected_max'].min(),
            'corrected_max_max': adj_df['corrected_max'].max(),
            
            'mean_max_correction': (adj_df['corrected_max'] - adj_df['original_max']).mean(),
            'max_correction_range': (adj_df['corrected_max'] - adj_df['original_max']).min(),
            'max_correction_max': (adj_df['corrected_max'] - adj_df['original_max']).max(),
            
            'roi_max_mean': adj_df['roi_max'].mean(),
            'roi_max_std': adj_df['roi_max'].std(),
            'roi_max_min': adj_df['roi_max'].min(),
            'roi_max_max': adj_df['roi_max'].max(),
            
            'max_vs_roi_diff_mean': (adj_df['corrected_max'] - adj_df['roi_max']).mean(),
            'max_vs_roi_diff_std': (adj_df['corrected_max'] - adj_df['roi_max']).std(),
        }
        
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(focused_export_dir, 'max_roi_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"    ✓ Summary statistics: {summary_path}")
    
    # =========================================================================
    # 2. RAW VS ADJUSTED COMPARISON
    # =========================================================================
    if not raw_df.empty and not adj_df.empty:
        print(f"  Exporting Raw vs Adjusted comparison...")
        
        comparison_data = []
        min_len = min(len(raw_df), len(adj_df))
        
        for i in range(min_len):
            raw_row = raw_df.iloc[i]
            adj_row = adj_df.iloc[i]
            
            comp_row = {
                'frame': i,
                'time_s': raw_row['time_s'],
                'raw_max': raw_row['corrected_max'],
                'arduino_max': adj_row['original_max'],
                'corrected_max': adj_row['corrected_max'],
                'arduino_correction': adj_row['original_max'] - raw_row['corrected_max'],
                'emissivity_correction': adj_row['corrected_max'] - adj_row['original_max'],
                'total_correction': adj_row['corrected_max'] - raw_row['corrected_max'],
                'raw_roi_max': raw_row['roi_max'],
                'adjusted_roi_max': adj_row['roi_max'],
            }
            
            comparison_data.append(comp_row)
        
        comparison_df = pd.DataFrame(comparison_data)
        if not comparison_df.empty:
            start_raw_max = comparison_df['raw_max'].iloc[0]
            comparison_df['normalized_raw_max'] = comparison_df['raw_max'] - start_raw_max
            start_corrected_max = comparison_df['corrected_max'].iloc[0]
            comparison_df['normalized_corrected_max'] = comparison_df['corrected_max'] - start_corrected_max
            start_raw_roi_max = comparison_df['raw_roi_max'].iloc[0]
            comparison_df['normalized_raw_roi_max'] = comparison_df['raw_roi_max'] - start_raw_roi_max
            start_adjusted_roi_max = comparison_df['adjusted_roi_max'].iloc[0]
            comparison_df['normalized_adjusted_roi_max'] = comparison_df['adjusted_roi_max'] - start_adjusted_roi_max
        comparison_path = os.path.join(focused_export_dir, 'raw_vs_adjusted_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"    ✓ Comparison data: {comparison_path}")
    
    print(f"  ✓ All focused data exported successfully!")
    return focused_export_dir

def create_comprehensive_report(raw_df, adj_df, filepath, output_dir):
    """Create a comprehensive report focusing on Max Pixel and ROI analysis."""
    print(f"\n[4] Creating comprehensive report...")
    
    report_path = os.path.join(output_dir, 'max_roi_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MAX PIXEL & ROI THERMAL ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source File: {os.path.basename(filepath)}\n")
        f.write(f"Total Frames: {len(adj_df) if not adj_df.empty else 0}\n")
        f.write(f"Duration: {adj_df['time_s'].iloc[-1] if not adj_df.empty else 0:.1f} seconds\n")
        f.write(f"Sampling Interval: {SAMPLING_RATE} seconds\n")
        f.write(f"Mouse Emissivity (ε): {MOUSE_EMISSIVITY}\n")
        f.write(f"ROI Size: {ROI_SIZE}x{ROI_SIZE} pixels\n\n")
        
        if not adj_df.empty:
            # Max Pixel Analysis
            f.write("MAX PIXEL (BRIGHTEST PIXEL) ANALYSIS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Arduino Max Pixel (ε=1.0 assumed):\n")
            f.write(f"  Mean: {adj_df['original_max'].mean():.2f} °C\n")
            f.write(f"  Std:  {adj_df['original_max'].std():.2f} °C\n")
            f.write(f"  Range: {adj_df['original_max'].min():.1f} to {adj_df['original_max'].max():.1f} °C\n\n")
            
            f.write(f"ε-Corrected Max Pixel (ε={MOUSE_EMISSIVITY}):\n")
            f.write(f"  Mean: {adj_df['corrected_max'].mean():.2f} °C\n")
            f.write(f"  Std:  {adj_df['corrected_max'].std():.2f} °C\n")
            f.write(f"  Range: {adj_df['corrected_max'].min():.1f} to {adj_df['corrected_max'].max():.1f} °C\n\n")
            
            max_correction = adj_df['corrected_max'] - adj_df['original_max']
            f.write(f"Max Pixel Correction (ε={MOUSE_EMISSIVITY}):\n")
            f.write(f"  Mean: {max_correction.mean():.3f} °C\n")
            f.write(f"  Std:  {max_correction.std():.3f} °C\n")
            f.write(f"  Range: {max_correction.min():.3f} to {max_correction.max():.3f} °C\n")
            f.write(f"  Expected: {CompleteThermalAnalyzer.calculate_expected_correction(adj_df['original_max'].mean(), MOUSE_EMISSIVITY):.3f} °C\n\n")
            
            # ROI Analysis
            f.write("ROI (3x3 REGION) ANALYSIS:\n")
            f.write("-"*40 + "\n")
            f.write(f"ROI Maximum Temperature:\n")
            f.write(f"  Mean: {adj_df['roi_max'].mean():.2f} °C\n")
            f.write(f"  Std:  {adj_df['roi_max'].std():.2f} °C\n")
            f.write(f"  Range: {adj_df['roi_max'].min():.1f} to {adj_df['roi_max'].max():.1f} °C\n\n")
            
            f.write(f"ROI Mean Temperature:\n")
            f.write(f"  Mean: {adj_df['roi_mean'].mean():.2f} °C\n")
            f.write(f"  Std:  {adj_df['roi_mean'].std():.2f} °C\n")
            f.write(f"  Range: {adj_df['roi_mean'].min():.1f} to {adj_df['roi_mean'].max():.1f} °C\n\n")
            
            # Comparison
            f.write("MAX PIXEL vs ROI COMPARISON:\n")
            f.write("-"*40 + "\n")
            max_vs_roi = adj_df['corrected_max'] - adj_df['roi_max']
            f.write(f"Max Pixel vs ROI Max Difference:\n")
            f.write(f"  Mean: {max_vs_roi.mean():.3f} °C\n")
            f.write(f"  Std:  {max_vs_roi.std():.3f} °C\n")
            f.write(f"  Range: {max_vs_roi.min():.3f} to {max_vs_roi.max():.3f} °C\n")
            f.write(f"  % frames where Max > ROI: {(max_vs_roi > 0).sum()/len(max_vs_roi)*100:.1f}%\n\n")
            
            # Raw vs Adjusted comparison
            if not raw_df.empty:
                f.write("RAW SENSOR vs ARDUINO ADJUSTED:\n")
                f.write("-"*40 + "\n")
                raw_vs_adj = adj_df['corrected_max'].iloc[:len(raw_df)] - raw_df['corrected_max']
                f.write(f"Total Correction (Corrected - Raw):\n")
                f.write(f"  Mean: {raw_vs_adj.mean():.2f} °C\n")
                f.write(f"  Range: {raw_vs_adj.min():.1f} to {raw_vs_adj.max():.1f} °C\n\n")
            
            # Key Insights
            f.write("KEY INSIGHTS:\n")
            f.write("-"*40 + "\n")
            
            if max_correction.mean() > 0:
                f.write("✓ CORRECTION VALID: Max pixel temperatures increased as expected\n")
                f.write("  The sensor was underestimating temperatures due to ε mismatch\n")
            else:
                f.write("⚠️ WARNING: Max pixel correction is negative\n")
                f.write("  Check if Arduino already applied emissivity correction\n")
            
            if max_vs_roi.mean() > 0.5:
                f.write("⚠️ NOTE: Max pixel significantly hotter than ROI max\n")
                f.write("  Suggests localized hot spot rather than uniform heating\n")
            elif max_vs_roi.mean() < 0.1:
                f.write("✓ NOTE: Max pixel and ROI max are very close\n")
                f.write("  Suggests uniform heating in the 3x3 region\n")
            
            f.write(f"\nMax pixel movement: Check movement_tracking.png for spatial stability\n")
            f.write(f"3D visualizations: View thermal surface plots in 3d_visualizations/\n")
        
        f.write("\nRECOMMENDATIONS FOR ANALYSIS:\n")
        f.write("-"*40 + "\n")
        f.write("1. Use 'corrected_max' for single hottest point analysis\n")
        f.write("2. Use 'roi_max' for more robust regional analysis (3x3 pixels)\n")
        f.write("3. For temporal analysis, use smoothed versions if available\n")
        f.write("4. Check 3D visualizations for spatial temperature patterns\n")
        f.write("5. Monitor max pixel movement for experimental stability\n")
        f.write("6. Compare with ambient temperature for delta-T analysis\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Analysis complete. All plots and data available in output directory.\n")
        f.write("="*80 + "\n")
    
    print(f"  ✓ Created comprehensive report: {report_path}")
    return report_path

def main():
    """Main analysis workflow focused on Max Pixel and ROI temperatures."""
    print("\n" + "="*80)
    print("MAX PIXEL & ROI FOCUSED THERMAL ANALYSIS")
    print("Focus: Brightest Pixel and 3x3 ROI temperatures (not means)")
    print("Includes: 3D visualizations, movement tracking, time series")
    print("="*80)
    
    # Select file
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    filepath = filedialog.askopenfilename(
        title="Select Arduino Thermal Data File",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()
    
    if not filepath:
        print("No file selected. Exiting.")
        return
    
    # User input for multiple animals
    num_animals = int(input("Number of animals (1,2,4): "))
    if num_animals not in [1,2,4]:
        print("Invalid choice, defaulting to 1 animal.")
        num_animals = 1
    
    division_type = None
    if num_animals == 2:
        division_type = input("Divide horizontally (h) or vertically (v): ").lower()
        if division_type not in ['h', 'v']:
            division_type = 'h'
    elif num_animals == 4:
        division_type = input("Divide into quadrants (q) or horizontal divisions (h): ").lower()
        if division_type not in ['q', 'h']:
            division_type = 'q'
    
    # Define regions
    if num_animals == 1:
        regions = [('full', slice(0,4), slice(0,16))]
    elif num_animals == 2:
        if division_type == 'h':
            regions = [('Left', slice(0,4), slice(0,8)), ('Right', slice(0,4), slice(8,16))]
        else:
            regions = [('Top', slice(0,2), slice(0,16)), ('Bottom', slice(2,4), slice(0,16))]
    elif num_animals == 4:
        if division_type == 'q':
            regions = [('TL', slice(0,2), slice(0,8)), ('TR', slice(0,2), slice(8,16)),
                       ('BL', slice(2,4), slice(0,8)), ('BR', slice(2,4), slice(8,16))]
        else:  # horizontal divisions
            regions = [('L1', slice(0,4), slice(0,4)), ('L2', slice(0,4), slice(4,8)),
                       ('L3', slice(0,4), slice(8,12)), ('L4', slice(0,4), slice(12,16))]
    
    # Create output directory with timestamp
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(filepath), 
                             f"{base_name}_max_roi_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Analyze data with focus on Max Pixel and ROI
    print(f"\n[1] Analyzing data with focus on Max Pixel and ROI temperatures...")
    processed = analyze_with_focus_on_max_and_roi(filepath, regions=regions)
    
    if processed is None:
        print("ERROR: Could not process data.")
        return
    
    is_multi = isinstance(processed, dict) and len(processed) > 1
    if not is_multi:
        processed = {'full': processed}
    
    for reg_name, (raw_df, adj_df) in processed.items():
        if raw_df is None or adj_df is None:
            continue
        
        sub_output_dir = os.path.join(output_dir, reg_name) if is_multi else output_dir
        os.makedirs(sub_output_dir, exist_ok=True)
        
        suffixed_filepath = f"{filepath}_{reg_name}" if is_multi else filepath
        
        print(f"\nProcessing region: {reg_name}")
        print(f"  Adjusted Grid Statistics (FOCUS ON MAX VALUES):")
        print(f"    Frames: {len(adj_df)}")
        print(f"    Arduino Max Pixel: {adj_df['original_max'].mean():.1f}°C "
              f"({adj_df['original_max'].min():.1f} to {adj_df['original_max'].max():.1f}°C)")
        print(f"    ε-Corrected Max Pixel: {adj_df['corrected_max'].mean():.1f}°C "
              f"({adj_df['corrected_max'].min():.1f} to {adj_df['corrected_max'].max():.1f}°C)")
        print(f"    ROI Max (3x3): {adj_df['roi_max'].mean():.1f}°C "
              f"({adj_df['roi_max'].min():.1f} to {adj_df['roi_max'].max():.1f}°C)")
        print(f"    Mean correction: {(adj_df['corrected_max'] - adj_df['original_max']).mean():.3f}°C")
        print(f"    Max vs ROI difference: {(adj_df['corrected_max'] - adj_df['roi_max']).mean():.3f}°C")
        
        # Create focused plots
        print(f"\n[2] Creating focused plots for Max Pixel and ROI...")
        create_max_and_roi_plots(
            raw_df, adj_df, suffixed_filepath, sub_output_dir
        )
        
        # Export focused results
        print(f"\n[3] Exporting focused data...")
        export_focused_results(raw_df, adj_df, suffixed_filepath, sub_output_dir)
        
        # Create comprehensive report
        print(f"\n[4] Creating comprehensive report...")
        create_comprehensive_report(raw_df, adj_df, suffixed_filepath, sub_output_dir)
    
    if is_multi:
        # Create combined folder
        combined_dir = os.path.join(output_dir, 'combined')
        os.makedirs(combined_dir, exist_ok=True)
        
        # Combined full-resolution CSV
        combined_df = pd.DataFrame()
        first_adj_df = list(processed.values())[0][1]
        combined_df['time_s'] = first_adj_df['time_s']
        
        for reg_name, (_, adj_df_reg) in processed.items():
            combined_df[f'corrected_max_{reg_name}'] = adj_df_reg['corrected_max']
            combined_df[f'roi_max_{reg_name}'] = adj_df_reg['roi_max']
        
        for reg_name in processed:
            col_max = f'corrected_max_{reg_name}'
            start = combined_df[col_max].iloc[0]
            combined_df[f'normalized_{col_max}'] = combined_df[col_max] - start
            col_roi = f'roi_max_{reg_name}'
            start = combined_df[col_roi].iloc[0]
            combined_df[f'normalized_{col_roi}'] = combined_df[col_roi] - start
        
        combined_csv_path = os.path.join(combined_dir, 'full_resolution_combined_temperatures.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\nCreated full-resolution combined CSV: {combined_csv_path}")
        
        # Compute downsampled data for combined plots
        chunk_size = max(1, round(DOWN_SAMPLE_PERIOD / SAMPLING_RATE))
        downsampled_times = []
        region_data = {}
        times = first_adj_df['time_s'].values
        for start in range(0, len(first_adj_df), chunk_size):
            end = start + chunk_size
            chunk_times = first_adj_df.iloc[start:end]['time_s']
            downsampled_times.append(np.median(chunk_times))
        downsampled_times = np.array(downsampled_times)
        
        for reg_name, (raw_df_reg, adj_df_reg) in processed.items():
            ds_original_max_med = []
            ds_original_max_std = []
            ds_corrected_max_med = []
            ds_corrected_max_std = []
            ds_roi_max_med = []
            ds_roi_max_std = []
            ds_raw_corrected_max_med = []
            ds_raw_corrected_max_std = []
            ds_raw_roi_max_med = []
            ds_raw_roi_max_std = []
            for start in range(0, len(adj_df_reg), chunk_size):
                end = start + chunk_size
                adj_chunk = adj_df_reg.iloc[start:end]
                raw_chunk = raw_df_reg.iloc[start:end]
                ds_original_max_med.append(np.median(adj_chunk['original_max']))
                ds_original_max_std.append(np.std(adj_chunk['original_max']))
                ds_corrected_max_med.append(np.median(adj_chunk['corrected_max']))
                ds_corrected_max_std.append(np.std(adj_chunk['corrected_max']))
                ds_roi_max_med.append(np.median(adj_chunk['roi_max']))
                ds_roi_max_std.append(np.std(adj_chunk['roi_max']))
                ds_raw_corrected_max_med.append(np.median(raw_chunk['corrected_max']))
                ds_raw_corrected_max_std.append(np.std(raw_chunk['corrected_max']))
                ds_raw_roi_max_med.append(np.median(raw_chunk['roi_max']))
                ds_raw_roi_max_std.append(np.std(raw_chunk['roi_max']))
            region_data[reg_name] = {
                'original_max_med': np.array(ds_original_max_med),
                'original_max_std': np.array(ds_original_max_std),
                'corrected_max_med': np.array(ds_corrected_max_med),
                'corrected_max_std': np.array(ds_corrected_max_std),
                'roi_max_med': np.array(ds_roi_max_med),
                'roi_max_std': np.array(ds_roi_max_std),
                'raw_corrected_max_med': np.array(ds_raw_corrected_max_med),
                'raw_corrected_max_std': np.array(ds_raw_corrected_max_std),
                'raw_roi_max_med': np.array(ds_raw_roi_max_med),
                'raw_roi_max_std': np.array(ds_raw_roi_max_std),
            }
            region_data[reg_name]['normalized_corrected_max_med'] = region_data[reg_name]['corrected_max_med'] - region_data[reg_name]['corrected_max_med'][0]
            region_data[reg_name]['normalized_roi_max_med'] = region_data[reg_name]['roi_max_med'] - region_data[reg_name]['roi_max_med'][0]
            region_data[reg_name]['normalized_raw_corrected_max_med'] = region_data[reg_name]['raw_corrected_max_med'] - region_data[reg_name]['raw_corrected_max_med'][0]
            region_data[reg_name]['normalized_raw_roi_max_med'] = region_data[reg_name]['raw_roi_max_med'] - region_data[reg_name]['raw_roi_max_med'][0]
        
        # Downsampled combined CSV
        combined_ds_df = pd.DataFrame({'time_s': downsampled_times})
        for reg_name, data in region_data.items():
            combined_ds_df[f'corrected_max_{reg_name}'] = data['corrected_max_med']
            combined_ds_df[f'roi_max_{reg_name}'] = data['roi_max_med']
            combined_ds_df[f'normalized_corrected_max_{reg_name}'] = data['normalized_corrected_max_med']
            combined_ds_df[f'normalized_roi_max_{reg_name}'] = data['normalized_roi_max_med']
        combined_ds_csv_path = os.path.join(combined_dir, 'downsampled_combined_temperatures.csv')
        combined_ds_df.to_csv(combined_ds_csv_path, index=False)
        print(f"Created downsampled combined CSV: {combined_ds_csv_path}")
        
        # Combined max pixel vs roi time series
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        colors = ['r', 'g', 'b', 'm']  # for up to 4 regions
        color_idx = 0
        for reg_name, data in region_data.items():
            c = colors[color_idx % len(colors)]
            # Panel 1: Max Pixel
            axes[0].plot(downsampled_times, data['normalized_corrected_max_med'], f'{c}-', label=f'{reg_name} Normalized Corrected Max')
            axes[0].fill_between(downsampled_times, data['normalized_corrected_max_med'] - data['corrected_max_std'], 
                                 data['normalized_corrected_max_med'] + data['corrected_max_std'], alpha=0.1, color=c)
            # Panel 2: ROI Max
            axes[1].plot(downsampled_times, data['normalized_roi_max_med'], f'{c}-', label=f'{reg_name} Normalized ROI Max')
            axes[1].fill_between(downsampled_times, data['normalized_roi_max_med'] - data['roi_max_std'], 
                                 data['normalized_roi_max_med'] + data['roi_max_std'], alpha=0.1, color=c)
            # Panel 3: Comparison Max vs ROI per region
            diff = data['normalized_corrected_max_med'] - data['normalized_roi_max_med']
            axes[2].plot(downsampled_times, data['normalized_corrected_max_med'], f'{c}-', label=f'{reg_name} Normalized Max')
            axes[2].plot(downsampled_times, data['normalized_roi_max_med'], f'{c}--', label=f'{reg_name} Normalized ROI Max')
            axes[2].plot(downsampled_times, diff, f'{c}:', label=f'{reg_name} Diff')
            color_idx += 1
        for i, ax in enumerate(axes):
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Normalized Temperature (°C)')
        axes[0].set_title('MAX PIXEL TEMPERATURE: Brightest (Hottest) Pixel')
        axes[1].set_title('ROI (3x3 REGION) MAXIMUM TEMPERATURE')
        axes[2].set_title('MAX PIXEL vs ROI MAX COMPARISON')
        plt.suptitle('Combined FOCUSED TEMPERATURE ANALYSIS: MAX PIXEL & ROI', fontsize=16, fontweight='bold')
        plt.tight_layout()
        combined_ts_path = os.path.join(combined_dir, 'combined_max_pixel_vs_roi_time_series.png')
        plt.savefig(combined_ts_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Created combined time series plot: {combined_ts_path}")
        
        # Combined detailed max roi analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        color_idx = 0
        for reg_name, data in region_data.items():
            c = colors[color_idx % len(colors)]
            # Ax0: Max Pixel Corrected
            axes[0].plot(downsampled_times, data['normalized_corrected_max_med'], f'{c}-', label=f'{reg_name} Normalized Corrected')
            axes[0].fill_between(downsampled_times, data['normalized_corrected_max_med'] - data['corrected_max_std'], 
                                 data['normalized_corrected_max_med'] + data['corrected_max_std'], alpha=0.1, color=c)
            # Ax1: Correction magnitude
            correction = data['corrected_max_med'] - data['original_max_med']
            axes[1].plot(downsampled_times, correction, f'{c}-', label=reg_name)
            # Ax2: ROI Max
            axes[2].plot(downsampled_times, data['normalized_roi_max_med'], f'{c}-', label=reg_name)
            axes[2].fill_between(downsampled_times, data['normalized_roi_max_med'] - data['roi_max_std'], 
                                 data['normalized_roi_max_med'] + data['roi_max_std'], alpha=0.1, color=c)
            # Ax3: Rate of change (using corrected_max)
            rate = np.gradient(data['normalized_corrected_max_med'], downsampled_times) * 60
            axes[3].plot(downsampled_times, rate, f'{c}-', label=reg_name)
            color_idx += 1
        for ax in axes:
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Normalized Temperature (°C)' if ax == axes[0] or ax == axes[2] else 'Correction (°C)' if ax == axes[1] else 'Rate (°C/min)')
        axes[0].set_title('MAX PIXEL: ε-Corrected')
        axes[1].set_title('Max Pixel Correction')
        axes[2].set_title('ROI MAXIMUM')
        axes[3].set_title('Max Pixel Rate of Change')
        plt.suptitle('Combined DETAILED MAX PIXEL & ROI ANALYSIS', fontsize=16, fontweight='bold')
        plt.tight_layout()
        combined_detail_path = os.path.join(combined_dir, 'combined_detailed_max_roi_analysis.png')
        plt.savefig(combined_detail_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Created combined detailed plot: {combined_detail_path}")
        
        # Combined correction chain
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        color_idx = 0
        for reg_name, data in region_data.items():
            c = colors[color_idx % len(colors)]
            # Ax0: Max Pixel comparison
            axes[0].plot(downsampled_times, data['normalized_raw_corrected_max_med'], f'{c}:', label=f'{reg_name} Normalized Raw Max')
            axes[0].fill_between(downsampled_times, data['normalized_raw_corrected_max_med'] - data['raw_corrected_max_std'], 
                                 data['normalized_raw_corrected_max_med'] + data['raw_corrected_max_std'], alpha=0.1, color=c)
            axes[0].plot(downsampled_times, data['normalized_corrected_max_med'], f'{c}-', label=f'{reg_name} Normalized Corrected Max')
            axes[0].fill_between(downsampled_times, data['normalized_corrected_max_med'] - data['corrected_max_std'], 
                                 data['normalized_corrected_max_med'] + data['corrected_max_std'], alpha=0.1, color=c)
            # Ax1: ROI Max comparison
            axes[1].plot(downsampled_times, data['normalized_raw_roi_max_med'], f'{c}--', label=f'{reg_name} Normalized Raw ROI Max')
            axes[1].fill_between(downsampled_times, data['normalized_raw_roi_max_med'] - data['raw_roi_max_std'], 
                                 data['normalized_raw_roi_max_med'] + data['raw_roi_max_std'], alpha=0.1, color=c)
            axes[1].plot(downsampled_times, data['normalized_roi_max_med'], f'{c}-', label=f'{reg_name} Normalized Adjusted ROI Max')
            axes[1].fill_between(downsampled_times, data['normalized_roi_max_med'] - data['roi_max_std'], 
                                 data['normalized_roi_max_med'] + data['roi_max_std'], alpha=0.1, color=c)
            # Ax2: Correction chain
            total_corr = data['corrected_max_med'] - data['raw_corrected_max_med']
            axes[2].plot(downsampled_times, total_corr, f'{c}-', label=f'{reg_name} Total Corr')
            color_idx += 1
        for ax in axes:
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Normalized Temperature (°C)' if ax != axes[2] else 'Correction (°C)')
        axes[0].set_title('MAX PIXEL: Raw → ε-Corrected')
        axes[1].set_title('ROI MAX (3x3): Raw vs Adjusted')
        axes[2].set_title('CORRECTION CHAIN ANALYSIS')
        plt.suptitle('Combined COMPLETE CORRECTION ANALYSIS: MAX PIXEL FOCUS', fontsize=16, fontweight='bold')
        plt.tight_layout()
        combined_comparison_path = os.path.join(combined_dir, 'combined_max_pixel_correction_chain.png')
        plt.savefig(combined_comparison_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Created combined correction chain plot: {combined_comparison_path}")
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE - MAX PIXEL & ROI FOCUS!")
    print("="*80)
    print(f"\nOUTPUT DIRECTORY STRUCTURE:")
    print(f"{output_dir}/")
    if is_multi:
        for reg_name in processed:
            print(f"├── {reg_name}/  (per-animal data, plots, report)")
        print(f"└── combined/  (combined CSVs and line graphs)")
    else:
        print(f"├── plots/")
        print(f"│   ├── max_roi_focused/")
        print(f"│   │   ├── max_pixel_vs_roi_time_series.png")
        print(f"│   │   ├── detailed_max_roi_analysis.png")
        print(f"│   │   └── max_pixel_movement_tracking.png")
        print(f"│   ├── 3d_visualizations/")
        print(f"│   │   └── 3d_visualization_frame_*.png (3D surface plots)")
        print(f"│   └── comparison_analysis/")
        print(f"│       └── max_pixel_correction_chain.png")
        print(f"├── exports/")
        print(f"│   └── max_roi_focused_data/")
        print(f"│       ├── max_pixel_roi_focused_data.csv")
        print(f"│       ├── max_roi_summary.csv")
        print(f"│       └── raw_vs_adjusted_comparison.csv")
        print(f"└── max_roi_analysis_report.txt")
    
    print(f"\nKEY FINDINGS (for full or first region):")
    adj_df = list(processed.values())[0][1]
    print(f"  1. Max Pixel (brightest): {adj_df['corrected_max'].mean():.1f}°C")
    print(f"  2. ROI Max (3x3 region): {adj_df['roi_max'].mean():.1f}°C")
    print(f"  3. Difference (Max - ROI): {(adj_df['corrected_max'] - adj_df['roi_max']).mean():.3f}°C")
    print(f"  4. ε-correction applied: {(adj_df['corrected_max'] - adj_df['original_max']).mean():.3f}°C")
    
    if (adj_df['corrected_max'] - adj_df['original_max']).mean() > 0:
        print(f"  ✓ Correction is POSITIVE as expected for mouse emissivity")
    else:
        print(f"  ⚠️  Check ε value - correction should be positive")
    
    print(f"\nCheck the 3D visualizations for spatial thermal patterns!")
    print(f"="*80)

if __name__ == "__main__":
    main()