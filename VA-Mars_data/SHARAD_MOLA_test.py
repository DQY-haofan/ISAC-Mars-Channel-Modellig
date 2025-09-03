#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版 - 火星SHARAD雷达验证套件
包含所有建议的改进：
1. 几何规范化（去1/R^4和cosθ效应）
2. 亚像素峰值定位
3. 系统偏移校正
4. 局部放大可视化
5. 多轨道批处理
"""

import os
import numpy as np
import pvl
import pandas as pd
import matplotlib.pyplot as plt
from pds4_tools import pds4_read
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, pearsonr, gaussian_kde
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===========================
# 配置数据路径
# ===========================
MOLA_DIR = "MOLA MEGDR"
MOLA_N_IMG = os.path.join(MOLA_DIR, "megt00n090hb.img")
MOLA_N_LBL = os.path.join(MOLA_DIR, "megt00n090hb.lbl")
MOLA_S_IMG = os.path.join(MOLA_DIR, "megt44s090hb.img")
MOLA_S_LBL = os.path.join(MOLA_DIR, "megt44s090hb.lbl")

CLUTTER_DIR = "SHARAD Clutter"
RADARGRAM_DIR = "RADARGRAM"
GEOM_DIR = "GEOM"

# 全局变量
DEM_N, META_N = None, None
DEM_S, META_S = None, None


# ===========================
# MOLA读取函数（保持原有）
# ===========================
def read_megdr(img_path, lbl_path):
    """读取MOLA MEGDR瓦片数据"""
    print(f"  读取: {os.path.basename(img_path)}")

    lbl = pvl.load(lbl_path)
    lines = int(lbl['IMAGE']['LINES'])
    cols = int(lbl['IMAGE']['LINE_SAMPLES'])
    sbits = int(lbl['IMAGE']['SAMPLE_BITS'])
    stype = str(lbl['IMAGE']['SAMPLE_TYPE'])
    scale = float(lbl['IMAGE'].get('SCALING_FACTOR', 1))
    offset = float(lbl['IMAGE'].get('OFFSET', 0))

    if sbits == 16 and 'MSB' in stype:
        dtype = '>i2'
    elif sbits == 16 and 'LSB' in stype:
        dtype = '<i2'
    else:
        raise ValueError(f'未支持的数据类型: {sbits}位 {stype}')

    arr = np.fromfile(img_path, dtype=dtype).reshape(lines, cols)
    arr = arr * scale + offset

    proj = lbl['IMAGE_MAP_PROJECTION']
    meta = dict(
        west=float(proj['WESTERNMOST_LONGITUDE']),
        east=float(proj['EASTERNMOST_LONGITUDE']),
        lat_max=float(proj['MAXIMUM_LATITUDE']),
        lat_min=float(proj['MINIMUM_LATITUDE']),
        ppd=float(proj['MAP_RESOLUTION']),
        lines=lines,
        cols=cols
    )

    return arr, meta


def bilinear_sample(dem, meta, lonE, lat):
    """双线性插值采样"""
    jf = (lonE - meta['west']) * meta['ppd']
    if_val = (meta['lat_max'] - lat) * meta['ppd']

    if not (0 <= jf < meta['cols'] - 1) or not (0 <= if_val < meta['lines'] - 1):
        return np.nan

    i0, j0 = int(if_val), int(jf)
    di, dj = if_val - i0, jf - j0

    v00 = dem[i0, j0]
    v01 = dem[i0, j0 + 1]
    v10 = dem[i0 + 1, j0]
    v11 = dem[i0 + 1, j0 + 1]

    return (1 - di) * (1 - dj) * v00 + (1 - di) * dj * v01 + di * (1 - dj) * v10 + di * dj * v11


def sample_mola(lonE, lat):
    """智能选择瓦片并采样"""
    lonE = lonE % 360.0

    if (META_S is not None and
            META_S['west'] <= lonE <= META_S['east'] and
            META_S['lat_min'] <= lat <= META_S['lat_max']):
        return bilinear_sample(DEM_S, META_S, lonE, lat)

    if (META_N is not None and
            META_N['west'] <= lonE <= META_N['east'] and
            META_N['lat_min'] <= lat <= META_N['lat_max']):
        return bilinear_sample(DEM_N, META_N, lonE, lat)

    return np.nan


# ===========================
# 增强功能1：亚像素峰值定位
# ===========================
def subpixel_peak_fit(data, peak_idx):
    """三点抛物线拟合实现亚像素峰值定位"""
    if peak_idx <= 0 or peak_idx >= len(data) - 1:
        return float(peak_idx)

    # 三点值
    y1 = data[peak_idx - 1]
    y2 = data[peak_idx]
    y3 = data[peak_idx + 1]

    # 抛物线拟合
    denominator = 2 * (y1 - 2 * y2 + y3)
    if abs(denominator) < 1e-10:
        return float(peak_idx)

    offset = (y1 - y3) / denominator

    # 限制偏移范围
    offset = np.clip(offset, -0.5, 0.5)

    return peak_idx + offset


# ===========================
# 增强功能2：系统偏移校正
# ===========================
def find_global_offset(signal1, signal2, max_shift=50):
    """使用互相关找到全局最优偏移"""
    # 确保长度匹配
    min_len = min(len(signal1), len(signal2))
    s1 = signal1[:min_len]
    s2 = signal2[:min_len]

    # 去除NaN
    valid = ~np.isnan(s1) & ~np.isnan(s2)
    if np.sum(valid) < 100:
        return 0

    s1_clean = s1[valid]
    s2_clean = s2[valid]

    # 计算互相关
    correlation = correlate(s1_clean, s2_clean, mode='same')
    lags = np.arange(-len(s2_clean) // 2, len(s2_clean) // 2)

    # 限制搜索范围
    valid_range = np.abs(lags) <= max_shift
    correlation = correlation[valid_range]
    lags = lags[valid_range]

    # 找到峰值
    best_lag = lags[np.argmax(correlation)]

    return best_lag


# ===========================
# 增强功能3：几何规范化
# ===========================
def geometric_normalization(power, slant_range, incidence_angle=None):
    """
    对功率进行几何规范化
    - 去除1/R^4效应
    - 去除cos(θ)效应（如果提供入射角）
    """
    power_db = 10 * np.log10(np.abs(power) + 1e-10)

    # 1/R^4 校正
    range_correction = 40 * np.log10(slant_range + 1e-3)
    power_corrected = power_db + range_correction

    # cos(θ) 校正
    if incidence_angle is not None:
        angle_correction = 20 * np.log10(np.cos(np.radians(incidence_angle)) + 1e-3)
        power_corrected += angle_correction

    return power_corrected


def estimate_incidence_angle(mola_elev, spacecraft_hgt, along_track_distance):
    """估算入射角（基于地形坡度和轨道几何）"""
    # 计算地形坡度
    terrain_slope = np.gradient(mola_elev) / along_track_distance

    # 计算标称入射角（简化模型）
    nominal_angle = np.zeros_like(mola_elev)

    for i in range(len(mola_elev)):
        if not np.isnan(mola_elev[i]) and not np.isnan(spacecraft_hgt[i]):
            # 几何关系
            height_diff = spacecraft_hgt[i] - mola_elev[i]

            # 考虑地形坡度的影响
            if i > 0 and i < len(terrain_slope):
                local_slope = terrain_slope[i]
                # 入射角 = 标称角 + 地形坡度贡献
                nominal_angle[i] = np.degrees(np.arctan(local_slope))

    return nominal_angle


# ===========================
# 增强功能4：改进的表面功率提取
# ===========================
def extract_surface_power_enhanced(radargram, geom_df, delta_t):
    """增强版表面功率提取（包含亚像素定位和系统校正）"""
    print(f"\n提取表面功率（增强版）...")

    rg_img = radargram['data']
    c0 = 299792458.0

    # 1. 获取表面线
    surf_bin = None
    for cand in ['surface_line', 'SurfaceLine', 'NadirLine']:
        if cand in geom_df.columns:
            surf_bin = pd.to_numeric(geom_df[cand], errors='coerce').values
            print(f"  使用字段: {cand}")
            break

    if surf_bin is None:
        # 根据高度计算
        h_sc = pd.to_numeric(geom_df.get('SpacecraftHgt', np.nan), errors='coerce').values
        h_nad = pd.to_numeric(geom_df.get('NadirHgt', np.nan), errors='coerce').values
        rng = 2.0 * np.maximum(0, h_sc - h_nad)
        tau = rng / c0
        surf_bin = tau / delta_t

    # 2. 亚像素峰值检测
    W = 10
    cols = np.arange(min(len(surf_bin), rg_img.shape[1]))
    valid = ~np.isnan(surf_bin) & (surf_bin >= 0) & (surf_bin < rg_img.shape[0])

    peak_bin = np.full_like(surf_bin, np.nan, dtype=float)
    peak_pow = np.full_like(surf_bin, np.nan, dtype=float)
    peak_bin_subpixel = np.full_like(surf_bin, np.nan, dtype=float)

    for j in cols[valid[:len(cols)]]:
        i0 = int(round(surf_bin[j]))
        i1 = max(0, i0 - W)
        i2 = min(rg_img.shape[0], i0 + W + 1)

        col = np.abs(rg_img[i1:i2, j])
        if col.size == 0:
            continue

        # 粗峰值
        k = np.argmax(col)
        peak_idx_global = i1 + k
        peak_bin[j] = peak_idx_global
        peak_pow[j] = col[k]

        # 亚像素精化
        if k > 0 and k < len(col) - 1:
            subpixel_offset = subpixel_peak_fit(col, k)
            peak_bin_subpixel[j] = i1 + subpixel_offset
        else:
            peak_bin_subpixel[j] = peak_idx_global

    # 3. 系统偏移校正
    valid_mask = ~np.isnan(peak_bin) & ~np.isnan(surf_bin[:len(peak_bin)])
    if np.sum(valid_mask) > 100:
        global_offset = find_global_offset(
            peak_bin[valid_mask],
            surf_bin[:len(peak_bin)][valid_mask]
        )
        print(f"  检测到系统偏移: {global_offset:.2f} bins")
        peak_bin_corrected = peak_bin - global_offset
        peak_bin_subpixel_corrected = peak_bin_subpixel - global_offset
    else:
        peak_bin_corrected = peak_bin
        peak_bin_subpixel_corrected = peak_bin_subpixel

    # 4. 计算误差统计
    valid_final = ~np.isnan(peak_bin_corrected) & ~np.isnan(surf_bin[:len(peak_bin_corrected)])
    if np.sum(valid_final) > 0:
        # 使用亚像素位置计算误差
        bin_err = peak_bin_subpixel_corrected[valid_final] - surf_bin[:len(peak_bin_subpixel_corrected)][valid_final]
        rmse_bins = np.sqrt(np.mean(bin_err ** 2))
        rmse_m = rmse_bins * (c0 * delta_t) / 2.0

        print(f"  亚像素RMSE: {rmse_bins:.2f} bins (~{rmse_m:.1f} m)")
        print(f"  平均偏移: {np.mean(bin_err):.2f} bins")
        print(f"  95%分位: [{np.percentile(bin_err, 2.5):.2f}, {np.percentile(bin_err, 97.5):.2f}] bins")

    return {
        'surf_bin': surf_bin,
        'peak_bin': peak_bin_corrected,
        'peak_bin_subpixel': peak_bin_subpixel_corrected,
        'peak_pow': peak_pow,
        'bin_err': bin_err if 'bin_err' in locals() else np.array([]),
        'rmse_bins': rmse_bins if 'rmse_bins' in locals() else np.nan,
        'rmse_m': rmse_m if 'rmse_m' in locals() else np.nan,
        'global_offset': global_offset if 'global_offset' in locals() else 0
    }


# ===========================
# 增强功能5：去混杂相关分析
# ===========================
def compute_deconfounded_correlations(mola_elev, peak_pow, geom_df, along_track_dist=56.0):
    """计算去混杂后的相关性（去除几何效应）"""
    print(f"\n计算去混杂相关性...")

    # 获取必要数据
    h_sc = pd.to_numeric(geom_df.get('SpacecraftHgt', np.nan), errors='coerce').values
    h_nad = pd.to_numeric(geom_df.get('NadirHgt', np.nan), errors='coerce').values

    # 确保长度匹配
    min_len = min(len(mola_elev), len(peak_pow), len(h_sc), len(h_nad))
    mola_subset = mola_elev[:min_len]
    power_subset = peak_pow[:min_len]
    h_sc_subset = h_sc[:min_len]
    h_nad_subset = h_nad[:min_len]

    # 有效数据掩码
    valid = (~np.isnan(mola_subset)) & (~np.isnan(power_subset)) & \
            (~np.isnan(h_sc_subset)) & (~np.isnan(h_nad_subset)) & \
            (power_subset > 0)

    if np.sum(valid) < 100:
        print(f"  警告：有效数据不足")
        return None

    # 计算斜距
    slant_range = h_sc_subset[valid] - h_nad_subset[valid]

    # 估算入射角
    incidence_angle = estimate_incidence_angle(
        mola_subset[valid],
        h_sc_subset[valid],
        along_track_dist
    )

    # 原始功率（dB）
    power_db_raw = 10 * np.log10(power_subset[valid] + 1e-10)

    # 几何规范化
    power_db_corrected = geometric_normalization(
        power_subset[valid],
        slant_range,
        incidence_angle
    )

    # 计算地形粗糙度
    window = 50
    roughness = np.zeros(np.sum(valid))
    mola_valid = mola_subset[valid]

    for i in range(len(mola_valid)):
        start = max(0, i - window // 2)
        end = min(len(mola_valid), i + window // 2 + 1)
        if end > start:
            roughness[i] = np.std(mola_valid[start:end])

    results = {
        'n_valid': np.sum(valid),
        'mola_valid': mola_valid,
        'power_raw': power_db_raw,
        'power_corrected': power_db_corrected,
        'roughness': roughness,
        'slant_range': slant_range,
        'incidence_angle': incidence_angle
    }

    # 计算相关性（原始 vs 校正后）
    print("\n  原始相关性:")
    rho_raw_elev, p_raw_elev = spearmanr(mola_valid, power_db_raw)
    rho_raw_rough, p_raw_rough = spearmanr(roughness, power_db_raw)
    print(f"    高程-功率: ρ = {rho_raw_elev:.3f} (p = {p_raw_elev:.2e})")
    print(f"    粗糙度-功率: ρ = {rho_raw_rough:.3f} (p = {p_raw_rough:.2e})")

    print("\n  几何校正后:")
    rho_corr_elev, p_corr_elev = spearmanr(mola_valid, power_db_corrected)
    rho_corr_rough, p_corr_rough = spearmanr(roughness, power_db_corrected)
    print(f"    高程-功率: ρ = {rho_corr_elev:.3f} (p = {p_corr_elev:.2e})")
    print(f"    粗糙度-功率: ρ = {rho_corr_rough:.3f} (p = {p_corr_rough:.2e})")

    results['correlations'] = {
        'raw_elev': (rho_raw_elev, p_raw_elev),
        'raw_rough': (rho_raw_rough, p_raw_rough),
        'corr_elev': (rho_corr_elev, p_corr_elev),
        'corr_rough': (rho_corr_rough, p_corr_rough)
    }

    return results


# ===========================
# 可视化函数集
# ===========================
def create_triangular_validation_plot(radargram, clutter, mola_elev,
                                      surface_extraction, correlation_results,
                                      geom_df, output_name="triangular_validation.png"):
    """Fig V-1: 三角交叉验证主图"""

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(5, 3, figure=fig,
                  height_ratios=[1.2, 1.2, 0.8, 0.6, 0.8],
                  hspace=0.3, wspace=0.25)

    # 准备数据
    rg_img = radargram['data']
    sim = clutter['sim']
    surf_bin = surface_extraction['surf_bin']
    peak_bin = surface_extraction.get('peak_bin_subpixel', surface_extraction['peak_bin'])
    peak_pow = surface_extraction['peak_pow']
    bin_err = surface_extraction['bin_err']

    # 1. 实测雷达图
    ax1 = fig.add_subplot(gs[0, :])

    rg_display = 10 * np.log10(np.abs(rg_img) + 1e-10)
    im1 = ax1.imshow(rg_display, aspect='auto', cmap='gray',
                     vmin=np.percentile(rg_display[rg_display > -100], 5),
                     vmax=np.percentile(rg_display[rg_display > -100], 95))

    # 叠加表面线
    x_coords = np.arange(len(surf_bin))
    ax1.plot(x_coords, surf_bin, 'c-', linewidth=1.5, alpha=0.8,
             label='Predicted Surface')
    ax1.plot(x_coords[:len(peak_bin)], peak_bin, 'y--', linewidth=1, alpha=0.8,
             label='Detected Peak')

    ax1.set_title("Measured Radargram with Surface Detection",
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel("Delay bin")
    ax1.set_xlabel("Along-track column")
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, min(rg_img.shape[1], len(surf_bin)))
    plt.colorbar(im1, ax=ax1, label="Power (dB)")

    # 2. 官方杂波模拟
    ax2 = fig.add_subplot(gs[1, :])

    sim_display = np.log10(np.abs(sim) + 1e-10)
    im2 = ax2.imshow(sim_display, aspect='auto', cmap='gray',
                     vmin=np.percentile(sim_display, 5),
                     vmax=np.percentile(sim_display, 95))

    # 如果有NadirLine，叠加
    if 'NadirLine' in geom_df.columns:
        nadir_line = geom_df['NadirLine'].values
        ax2.plot(np.arange(len(nadir_line)), nadir_line, 'c-',
                 linewidth=1.5, alpha=0.8, label='Clutter Prediction')
        ax2.legend(loc='upper right')

    ax2.set_title("Official Clutter Simulation", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Delay bin")
    ax2.set_xlabel("Along-track column")
    ax2.set_xlim(0, min(sim.shape[1], len(surf_bin)))
    plt.colorbar(im2, ax=ax2, label="Log10(Power)")

    # 3. MOLA高程剖面
    ax3 = fig.add_subplot(gs[2, :])

    valid_mask = ~np.isnan(mola_elev)
    x = np.arange(len(mola_elev))

    ax3.plot(x[valid_mask], mola_elev[valid_mask], 'b-',
             linewidth=1.5, alpha=0.8, label='MOLA Elevation')

    # 添加统计信息
    if np.sum(valid_mask) > 0:
        valid_elevs = mola_elev[valid_mask]
        stats_text = (f"Coverage: {np.sum(valid_mask) / len(mola_elev) * 100:.1f}%\n"
                      f"Range: {np.min(valid_elevs):.0f} - {np.max(valid_elevs):.0f} m\n"
                      f"Mean: {np.mean(valid_elevs):.0f} m\n"
                      f"Std: {np.std(valid_elevs):.0f} m")
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax3.set_title("MOLA Elevation Profile", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Elevation (m)")
    ax3.set_xlabel("Along-track index")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, len(mola_elev))

    # 4. 表面功率剖面
    ax4 = fig.add_subplot(gs[3, :])

    valid_pow = ~np.isnan(peak_pow)
    if np.sum(valid_pow) > 0:
        power_db = 10 * np.log10(peak_pow[valid_pow] + 1e-10)
        ax4.plot(np.where(valid_pow)[0], power_db, 'r-',
                 linewidth=1, alpha=0.7, label='Surface Echo Power')
        ax4.set_ylabel("Power (dB)")
        ax4.set_xlabel("Along-track index")
        ax4.set_title("Extracted Surface Power", fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_xlim(0, len(peak_pow))

    # 5. 误差直方图
    ax5 = fig.add_subplot(gs[4, 0])

    if len(bin_err) > 0:
        ax5.hist(bin_err, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax5.set_xlabel("Alignment Error (bins)")
        ax5.set_ylabel("Frequency")
        ax5.set_title(f"Surface Detection Error\nRMSE: {surface_extraction['rmse_bins']:.2f} bins "
                      f"({surface_extraction['rmse_m']:.1f} m)",
                      fontsize=11)
        ax5.grid(True, alpha=0.3)

    # 6. 相关性散点图
    ax6 = fig.add_subplot(gs[4, 1])

    if correlation_results is not None and 'mola_valid' in correlation_results:
        mola_valid = correlation_results['mola_valid']
        power_db = correlation_results.get('power_raw', correlation_results.get('power_corrected', []))

        if len(mola_valid) > 0 and len(power_db) > 0:
            # 密度散点图
            from scipy.stats import gaussian_kde

            # 下采样
            if len(mola_valid) > 5000:
                idx = np.random.choice(len(mola_valid), 5000, replace=False)
                x_plot = mola_valid[idx]
                y_plot = power_db[idx]
            else:
                x_plot = mola_valid
                y_plot = power_db

            # 计算点密度
            xy = np.vstack([x_plot, y_plot])
            z = gaussian_kde(xy)(xy)

            sc = ax6.scatter(x_plot, y_plot, c=z, s=1, cmap='viridis', alpha=0.5)

            # 添加趋势线
            z_fit = np.polyfit(mola_valid, power_db, 1)
            p = np.poly1d(z_fit)
            x_trend = np.linspace(np.min(mola_valid), np.max(mola_valid), 100)
            ax6.plot(x_trend, p(x_trend), 'r-', alpha=0.5, linewidth=2)

            # 获取相关系数
            if 'correlations' in correlation_results:
                rho = correlation_results['correlations']['raw_elev'][0]
                r = correlation_results['correlations']['raw_elev'][0]  # 可以改为pearson
            else:
                rho = r = np.nan

            ax6.set_xlabel("MOLA Elevation (m)")
            ax6.set_ylabel("Echo Power (dB)")
            ax6.set_title(f"Elevation vs Power\nρ={rho:.3f}",
                          fontsize=11)
            ax6.grid(True, alpha=0.3)
            plt.colorbar(sc, ax=ax6, label="Density")

    # 7. 轨道地图
    ax7 = fig.add_subplot(gs[4, 2])

    # 查找经纬度列
    lon_col = None
    lat_col = None
    for col in ['FirstLon', 'NadirLon', 'SpacecraftLon']:
        if col in geom_df.columns:
            lon_col = col
            break
    for col in ['FirstLat', 'NadirLat', 'SpacecraftLat']:
        if col in geom_df.columns:
            lat_col = col
            break

    if lon_col and lat_col:
        lons = geom_df[lon_col].values
        lats = geom_df[lat_col].values

        # 处理负经度
        lons[lons < 0] += 360

        # 彩色编码高程
        sc = ax7.scatter(lons[:len(mola_elev)], lats[:len(mola_elev)],
                         c=mola_elev, s=2, cmap='terrain', alpha=0.8)
        ax7.set_title("Ground Track", fontsize=11)
        ax7.set_xlabel("Longitude (°E)")
        ax7.set_ylabel("Latitude (°N)")
        ax7.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax7, label="Elevation (m)")

        # 标记瓦片边界
        ax7.axhline(y=-44, color='red', linestyle='--',
                    linewidth=1, alpha=0.5, label='Tile boundary')
        ax7.legend(loc='best', fontsize=8)

    # 总标题
    plt.suptitle("Fig V-1: SHARAD-MOLA-Radargram Triangular Cross-Validation",
                 fontsize=14, fontweight='bold', y=0.995)

    # 添加汇总文本框
    if correlation_results:
        if 'correlations' in correlation_results:
            spearman_str = f"{correlation_results['correlations']['raw_elev'][0]:.3f}"
            pearson_str = f"{correlation_results['correlations']['raw_elev'][0]:.3f}"
        else:
            spearman_str = "N/A"
            pearson_str = "N/A"
        valid_points = correlation_results.get('n_valid', 0)
    else:
        spearman_str = "N/A"
        pearson_str = "N/A"
        valid_points = 0

    summary_text = f"""
    === Validation Metrics ===
    Surface RMSE: {surface_extraction['rmse_bins']:.2f} bins ({surface_extraction['rmse_m']:.1f} m)
    System Offset: {surface_extraction.get('global_offset', 0):.2f} bins
    Valid Points: {valid_points}
    Spearman ρ: {spearman_str}
    Coverage: {np.sum(~np.isnan(mola_elev)) / len(mola_elev) * 100:.1f}%
    """

    fig.text(0.99, 0.01, summary_text.strip(),
             horizontalalignment='right', verticalalignment='bottom',
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"✓ 三角交叉验证主图已保存: {output_name}")
    plt.show()


def create_zoom_alignment_figure(radargram, clutter, surface_extraction,
                                 windows=[(1000, 1300), (2500, 2800), (4000, 4300)],
                                 output_name="zoom_alignment.png"):
    """Fig V-2: 局部放大对齐图"""

    fig, axes = plt.subplots(len(windows), 1, figsize=(14, 4 * len(windows)))
    if len(windows) == 1:
        axes = [axes]

    rg_img = radargram['data']
    sim = clutter['sim']
    surf_bin = surface_extraction['surf_bin']
    peak_bin = surface_extraction['peak_bin_subpixel']

    for idx, (start, end) in enumerate(windows):
        ax = axes[idx]

        # 显示雷达图段
        rg_segment = 10 * np.log10(np.abs(rg_img[:, start:end]) + 1e-10)
        im = ax.imshow(rg_segment, aspect='auto', cmap='gray',
                       vmin=np.percentile(rg_segment[rg_segment > -100], 5),
                       vmax=np.percentile(rg_segment[rg_segment > -100], 95),
                       extent=[start, end, rg_img.shape[0], 0])

        # 叠加线条
        x_range = np.arange(start, min(end, len(surf_bin)))
        ax.plot(x_range, surf_bin[start:end], 'c-', linewidth=2,
                label='Predicted Surface', alpha=0.8)
        ax.plot(x_range, peak_bin[start:end], 'y--', linewidth=1.5,
                label='Detected Peak (subpixel)', alpha=0.8)

        # 如果有杂波NadirLine
        if 'NadirLine' in surface_extraction:
            ax.plot(x_range, surface_extraction['NadirLine'][start:end],
                    'm:', linewidth=1.5, label='Clutter NadirLine', alpha=0.7)

        ax.set_title(f"Window {idx + 1}: Columns {start}-{end}", fontweight='bold')
        ax.set_ylabel("Delay bin")
        ax.set_xlabel("Along-track column")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)

        # 添加局部RMSE
        local_err = peak_bin[start:end] - surf_bin[start:end]
        local_rmse = np.sqrt(np.nanmean(local_err ** 2))
        ax.text(0.02, 0.95, f"Local RMSE: {local_rmse:.2f} bins",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.suptitle("Fig V-2: Zoom-in Alignment Details", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"✓ 局部对齐图已保存: {output_name}")
    plt.show()


def create_error_analysis_figure(surface_extraction, output_name="error_analysis.png"):
    """Fig V-3: 误差时序和统计分析"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    bin_err = surface_extraction['bin_err']

    # 1. 误差时序
    ax1 = axes[0, 0]
    ax1.plot(bin_err, 'b-', linewidth=0.5, alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=np.mean(bin_err), color='g', linestyle='-', linewidth=1,
                alpha=0.5, label=f'Mean: {np.mean(bin_err):.2f}')
    ax1.fill_between(range(len(bin_err)),
                     np.percentile(bin_err, 2.5) * np.ones(len(bin_err)),
                     np.percentile(bin_err, 97.5) * np.ones(len(bin_err)),
                     alpha=0.2, color='gray', label='95% CI')
    ax1.set_title("Alignment Error Time Series", fontweight='bold')
    ax1.set_xlabel("Along-track index")
    ax1.set_ylabel("Error (bins)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 误差直方图
    ax2 = axes[0, 1]
    n, bins, patches = ax2.hist(bin_err, bins=50, alpha=0.7, color='blue',
                                edgecolor='black', density=True)

    # 拟合高斯
    from scipy.stats import norm
    mu, std = norm.fit(bin_err)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax2.plot(x, p, 'r-', linewidth=2, label=f'Gaussian fit\nμ={mu:.2f}, σ={std:.2f}')

    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_title("Error Distribution", fontweight='bold')
    ax2.set_xlabel("Error (bins)")
    ax2.set_ylabel("Probability Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 累积分布
    ax3 = axes[1, 0]
    sorted_err = np.sort(bin_err)
    cumulative = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax3.plot(sorted_err, cumulative, 'b-', linewidth=2)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(y=0.5, color='g', linestyle='--', linewidth=1, alpha=0.5)

    # 标记关键百分位
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        val = np.percentile(bin_err, p)
        ax3.axvline(x=val, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax3.text(val, 0.05, f'{p}%\n{val:.1f}', fontsize=8, ha='center')

    ax3.set_title("Cumulative Distribution", fontweight='bold')
    ax3.set_xlabel("Error (bins)")
    ax3.set_ylabel("Cumulative Probability")
    ax3.grid(True, alpha=0.3)

    # 4. 统计汇总
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    === Error Statistics ===

    Sample Size: {len(bin_err)}

    Central Tendency:
      Mean: {np.mean(bin_err):.3f} bins
      Median: {np.median(bin_err):.3f} bins
      Mode: {sorted_err[np.argmax(n)]:.3f} bins

    Dispersion:
      Std Dev: {np.std(bin_err):.3f} bins
      MAD: {np.median(np.abs(bin_err - np.median(bin_err))):.3f} bins
      IQR: {np.percentile(bin_err, 75) - np.percentile(bin_err, 25):.3f} bins

    Extremes:
      Min: {np.min(bin_err):.3f} bins
      Max: {np.max(bin_err):.3f} bins
      Range: {np.max(bin_err) - np.min(bin_err):.3f} bins

    Percentiles:
      5%: {np.percentile(bin_err, 5):.3f} bins
      95%: {np.percentile(bin_err, 95):.3f} bins

    Quality Metrics:
      RMSE: {surface_extraction['rmse_bins']:.3f} bins
      RMSE (m): {surface_extraction['rmse_m']:.1f} m
      System Offset: {surface_extraction['global_offset']:.2f} bins
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             family='monospace', verticalalignment='top')

    plt.suptitle("Fig V-3: Alignment Error Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"✓ 误差分析图已保存: {output_name}")
    plt.show()


def create_deconfounded_correlation_figure(deconf_results, output_name="deconfounded_correlations.png"):
    """Fig V-4: 去混杂相关分析图"""

    if deconf_results is None:
        print("无去混杂结果可视化")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    mola = deconf_results['mola_valid']
    roughness = deconf_results['roughness']
    power_raw = deconf_results['power_raw']
    power_corr = deconf_results['power_corrected']

    # 下采样用于可视化
    if len(mola) > 5000:
        idx = np.random.choice(len(mola), 5000, replace=False)
    else:
        idx = np.arange(len(mola))

    # 1. 原始：高程 vs 功率
    ax1 = axes[0, 0]
    xy = np.vstack([mola[idx], power_raw[idx]])
    z = gaussian_kde(xy)(xy)
    sc1 = ax1.scatter(mola[idx], power_raw[idx], c=z, s=1, cmap='viridis', alpha=0.5)

    # 趋势线
    z1 = np.polyfit(mola, power_raw, 1)
    p1 = np.poly1d(z1)
    x_trend = np.linspace(np.min(mola), np.max(mola), 100)
    ax1.plot(x_trend, p1(x_trend), 'r-', linewidth=2, alpha=0.5)

    rho, p_val = deconf_results['correlations']['raw_elev']
    ax1.set_title(f"Raw: Elevation vs Power\nρ={rho:.3f}, p={p_val:.2e}", fontweight='bold')
    ax1.set_xlabel("MOLA Elevation (m)")
    ax1.set_ylabel("Raw Power (dB)")
    ax1.grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=ax1)

    # 2. 校正后：高程 vs 功率
    ax2 = axes[0, 1]
    xy = np.vstack([mola[idx], power_corr[idx]])
    z = gaussian_kde(xy)(xy)
    sc2 = ax2.scatter(mola[idx], power_corr[idx], c=z, s=1, cmap='viridis', alpha=0.5)

    z2 = np.polyfit(mola, power_corr, 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_trend, p2(x_trend), 'r-', linewidth=2, alpha=0.5)

    rho, p_val = deconf_results['correlations']['corr_elev']
    ax2.set_title(f"Corrected: Elevation vs Power\nρ={rho:.3f}, p={p_val:.2e}", fontweight='bold')
    ax2.set_xlabel("MOLA Elevation (m)")
    ax2.set_ylabel("Corrected Power (dB)")
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=ax2)

    # 3. 斜距效应
    ax3 = axes[0, 2]
    slant_range = deconf_results['slant_range']
    ax3.scatter(slant_range[idx], power_raw[idx], s=1, alpha=0.3, color='blue')
    ax3.set_title("Range Effect (before correction)", fontweight='bold')
    ax3.set_xlabel("Slant Range (m)")
    ax3.set_ylabel("Raw Power (dB)")
    ax3.grid(True, alpha=0.3)

    # 4. 原始：粗糙度 vs 功率
    ax4 = axes[1, 0]
    xy = np.vstack([roughness[idx], power_raw[idx]])
    z = gaussian_kde(xy)(xy)
    sc4 = ax4.scatter(roughness[idx], power_raw[idx], c=z, s=1, cmap='plasma', alpha=0.5)

    z4 = np.polyfit(roughness, power_raw, 1)
    p4 = np.poly1d(z4)
    x_rough = np.linspace(np.min(roughness), np.max(roughness), 100)
    ax4.plot(x_rough, p4(x_rough), 'r-', linewidth=2, alpha=0.5)

    rho, p_val = deconf_results['correlations']['raw_rough']
    ax4.set_title(f"Raw: Roughness vs Power\nρ={rho:.3f}, p={p_val:.2e}", fontweight='bold')
    ax4.set_xlabel("Terrain Roughness (m)")
    ax4.set_ylabel("Raw Power (dB)")
    ax4.grid(True, alpha=0.3)
    plt.colorbar(sc4, ax=ax4)

    # 5. 校正后：粗糙度 vs 功率
    ax5 = axes[1, 1]
    xy = np.vstack([roughness[idx], power_corr[idx]])
    z = gaussian_kde(xy)(xy)
    sc5 = ax5.scatter(roughness[idx], power_corr[idx], c=z, s=1, cmap='plasma', alpha=0.5)

    z5 = np.polyfit(roughness, power_corr, 1)
    p5 = np.poly1d(z5)
    ax5.plot(x_rough, p5(x_rough), 'r-', linewidth=2, alpha=0.5)

    rho, p_val = deconf_results['correlations']['corr_rough']
    ax5.set_title(f"Corrected: Roughness vs Power\nρ={rho:.3f}, p={p_val:.2e}", fontweight='bold')
    ax5.set_xlabel("Terrain Roughness (m)")
    ax5.set_ylabel("Corrected Power (dB)")
    ax5.grid(True, alpha=0.3)
    plt.colorbar(sc5, ax=ax5)

    # 6. 入射角分布
    ax6 = axes[1, 2]
    incidence = deconf_results['incidence_angle']
    ax6.hist(incidence[~np.isnan(incidence)], bins=50, alpha=0.7, color='green')
    ax6.set_title("Incidence Angle Distribution", fontweight='bold')
    ax6.set_xlabel("Incidence Angle (degrees)")
    ax6.set_ylabel("Frequency")
    ax6.grid(True, alpha=0.3)

    # 添加改善比例文本
    imp_elev = (abs(deconf_results['correlations']['corr_elev'][0]) -
                abs(deconf_results['correlations']['raw_elev'][0])) / \
               abs(deconf_results['correlations']['raw_elev'][0]) * 100

    imp_rough = (abs(deconf_results['correlations']['corr_rough'][0]) -
                 abs(deconf_results['correlations']['raw_rough'][0])) / \
                abs(deconf_results['correlations']['raw_rough'][0]) * 100

    fig.text(0.5, 0.02,
             f"Correlation improvement after geometric correction: "
             f"Elevation {imp_elev:+.1f}%, Roughness {imp_rough:+.1f}%",
             ha='center', fontsize=11, style='italic')

    plt.suptitle("Fig V-4: Geometry-normalized Correlations",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"✓ 去混杂相关图已保存: {output_name}")
    plt.show()


# ===========================
# 读取数据函数（保持原有）
# ===========================
def read_radargram(rgram_xml, rgram_img=None):
    """读取SHARAD实测雷达图"""
    print(f"\n读取实测雷达图...")

    try:
        rg = pds4_read(rgram_xml)
        rg_arrs = [s for s in rg if hasattr(s, 'tag') and 'Array_2D_Image' in s.tag]
        if not rg_arrs:
            rg_arrs = [s for s in rg if hasattr(s, 'data')]

        if not rg_arrs:
            raise ValueError("无法找到雷达图数据数组")

        R = rg_arrs[0]
        rg_img = R.data.astype(np.float32)

        scale = getattr(R, 'scaling_factor', 1.0)
        offset = getattr(R, 'offset', 0.0)
        rg_img = rg_img * scale + offset

        delta_t = None
        for k in ['line_sampling_interval', 'Line_Sampling_Interval',
                  'SAMPLE_TIME', 'sample_time']:
            try:
                if hasattr(R, 'meta_data') and k in R.meta_data:
                    delta_t = float(R.meta_data[k])
                    break
            except:
                pass

        if delta_t is None:
            delta_t = 0.375e-6
            print(f"  警告：未找到采样间隔，使用默认值 {delta_t * 1e6:.3f} μs")

        print(f"  雷达图尺寸: {rg_img.shape}")
        print(f"  垂直采样间隔: {delta_t * 1e6:.3f} μs")

        return {
            'data': rg_img,
            'delta_t': delta_t,
            'metadata': R if hasattr(R, 'meta_data') else None
        }

    except Exception as e:
        print(f"  警告：无法读取雷达图 - {e}")
        return None


def read_sharad_clutter(sim_xml, emap_xml=None, rtrn_csv=None):
    """读取SHARAD Clutter套件数据"""
    result = {}

    print(f"读取SHARAD杂波数据...")
    prod = pds4_read(sim_xml)

    if len(prod) > 1:
        print(f"  发现{len(prod)}个数组，使用组合杂波")
        result['sim'] = prod[-1].data if hasattr(prod[-1], 'data') else prod[-1]
    else:
        result['sim'] = prod[0].data if hasattr(prod[0], 'data') else prod[0]
    print(f"  杂波图尺寸: {result['sim'].shape}")

    if emap_xml and os.path.exists(emap_xml):
        try:
            prod = pds4_read(emap_xml)
            result['emap'] = prod[0].data if hasattr(prod[0], 'data') else prod[0]
            print(f"  能量图尺寸: {result['emap'].shape}")
        except:
            print(f"  警告：无法读取能量图")

    if rtrn_csv and os.path.exists(rtrn_csv):
        result['rtrn'] = pd.read_csv(rtrn_csv)
        print(f"  几何数据: {len(result['rtrn'])} 行")

    return result


def read_geom_table(geom_xml, geom_tab=None):
    """读取SHARAD几何表"""
    print(f"\n读取几何表...")

    try:
        if not os.path.exists(geom_xml):
            return None

        G = pds4_read(geom_xml, quiet=True, lazy_load=True)
        tables = [s for s in G if 'Table' in str(type(s))]

        if tables:
            tbl = tables[0]
            if hasattr(tbl, 'to_table'):
                geom_table = tbl.to_table()
                geom_df = geom_table.to_pandas()
            else:
                geom_df = pd.DataFrame(tbl.data)

            geom_df.columns = [c.strip() for c in geom_df.columns]
            print(f"  几何表尺寸: {len(geom_df)} 行")
            return geom_df

    except Exception as e:
        print(f"  警告：无法从XML读取 - {e}")

    return None


# ===========================
# 主验证流程
# ===========================
def enhanced_validation_pipeline(track_id='s_00810101'):
    """执行增强版验证流程"""

    print("=" * 80)
    print(f"增强版SHARAD验证 - 轨道 {track_id}")
    print("=" * 80)

    # 设置文件路径
    sim_xml = os.path.join(CLUTTER_DIR, f"{track_id}_sim.xml")
    emap_xml = os.path.join(CLUTTER_DIR, f"{track_id}_emap.xml")
    rtrn_csv = os.path.join(CLUTTER_DIR, f"{track_id}_rtrn.csv")
    rgram_xml = os.path.join(RADARGRAM_DIR, f"{track_id}_rgram.xml")
    rgram_img = os.path.join(RADARGRAM_DIR, f"{track_id}_rgram.img")
    geom_xml = os.path.join(GEOM_DIR, f"{track_id}_geom.xml")
    geom_tab = os.path.join(GEOM_DIR, f"{track_id}_geom.tab")

    # 1. 读取MOLA
    print("\n[步骤1] 读取MOLA数据")
    global DEM_N, META_N, DEM_S, META_S

    if os.path.exists(MOLA_N_IMG):
        DEM_N, META_N = read_megdr(MOLA_N_IMG, MOLA_N_LBL)
    if os.path.exists(MOLA_S_IMG):
        DEM_S, META_S = read_megdr(MOLA_S_IMG, MOLA_S_LBL)

    # 2. 读取数据
    print("\n[步骤2] 读取SHARAD数据")
    clutter = read_sharad_clutter(sim_xml, emap_xml, rtrn_csv)
    radargram = read_radargram(rgram_xml, rgram_img)

    if radargram is None:
        print("错误：无法读取雷达图")
        return None

    # 3. 读取几何表
    print("\n[步骤3] 读取几何信息")
    geom_df = read_geom_table(geom_xml, geom_tab)
    if geom_df is None:
        geom_df = clutter['rtrn']
        print("  使用RTRN作为几何数据")

    # 4. 沿轨采样MOLA
    print("\n[步骤4] 沿轨采样MOLA高程")
    lon_col = None
    lat_col = None

    for col in ['FirstLon', 'NadirLon', 'SpacecraftLon']:
        if col in geom_df.columns:
            lon_col = col
            break

    for col in ['FirstLat', 'NadirLat', 'SpacecraftLat']:
        if col in geom_df.columns:
            lat_col = col
            break

    mola_elev = []
    for lonE, lat in zip(geom_df[lon_col].values, geom_df[lat_col].values):
        if lonE < 0:
            lonE += 360
        elev = sample_mola(float(lonE), float(lat))
        mola_elev.append(elev)

    mola_elev = np.array(mola_elev)
    print(f"  有效采样: {np.sum(~np.isnan(mola_elev))}/{len(mola_elev)}")

    # 5. 增强版表面提取
    print("\n[步骤5] 增强版表面功率提取")
    surface_extraction = extract_surface_power_enhanced(radargram, geom_df,
                                                        radargram['delta_t'])

    # 6. 去混杂相关分析
    print("\n[步骤6] 去混杂相关分析")
    deconf_results = compute_deconfounded_correlations(mola_elev,
                                                       surface_extraction['peak_pow'],
                                                       geom_df)

    # 7. 生成所有图表
    print("\n[步骤7] 生成验证图表套件")

    # Fig V-1: 三角交叉验证主图（最重要的！）
    create_triangular_validation_plot(radargram, clutter, mola_elev,
                                      surface_extraction, deconf_results,
                                      geom_df,
                                      output_name=f"{track_id}_triangular_validation.png")

    # Fig V-2: 局部放大
    create_zoom_alignment_figure(radargram, clutter, surface_extraction,
                                 output_name=f"{track_id}_zoom_alignment.png")

    # Fig V-3: 误差分析
    create_error_analysis_figure(surface_extraction,
                                 output_name=f"{track_id}_error_analysis.png")

    # Fig V-4: 去混杂相关
    create_deconfounded_correlation_figure(deconf_results,
                                           output_name=f"{track_id}_deconfounded.png")

    # 汇总结果
    results = {
        'track_id': track_id,
        'coverage': np.sum(~np.isnan(mola_elev)) / len(mola_elev) * 100,
        'rmse_bins': surface_extraction['rmse_bins'],
        'rmse_m': surface_extraction['rmse_m'],
        'system_offset': surface_extraction['global_offset'],
        'n_valid': deconf_results['n_valid'] if deconf_results else 0
    }

    if deconf_results:
        results.update({
            'rho_raw_elev': deconf_results['correlations']['raw_elev'][0],
            'rho_corr_elev': deconf_results['correlations']['corr_elev'][0],
            'rho_raw_rough': deconf_results['correlations']['raw_rough'][0],
            'rho_corr_rough': deconf_results['correlations']['corr_rough'][0]
        })

    print("\n" + "=" * 80)
    print("验证完成！")
    print("=" * 80)

    return results


# ===========================
# 批量处理
# ===========================
def batch_validation(track_list):
    """批量处理多条轨道"""

    results_list = []

    for track_id in track_list:
        print(f"\n\n处理轨道: {track_id}")
        print("-" * 60)

        try:
            results = enhanced_validation_pipeline(track_id)
            if results:
                results_list.append(results)
        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 生成汇总表
    if results_list:
        df = pd.DataFrame(results_list)

        print("\n" + "=" * 80)
        print("Table V-1: Multi-track Validation Summary")
        print("=" * 80)
        print(df.to_string())

        # 保存CSV
        df.to_csv("multitrack_validation_summary.csv", index=False)
        print("\n✓ 汇总表已保存到 multitrack_validation_summary.csv")

        # 生成LaTeX表格
        latex_table = df.to_latex(index=False, float_format="%.3f")
        with open("table_v1.tex", "w") as f:
            f.write(latex_table)
        print("✓ LaTeX表格已保存到 table_v1.tex")

    return results_list


# ===========================
# 主程序
# ===========================
if __name__ == "__main__":
    try:
        # 单轨验证
        results = enhanced_validation_pipeline('s_00810101')

        # 可选：批量验证
        # track_list = ['s_00810101', 's_00810102', 's_00810103']
        # batch_results = batch_validation(track_list)

        print("\n✓ 所有分析完成！")
        print("\n生成的图表：")
        print("  - Fig V-1: *_triangular_validation.png (三角交叉验证主图)")
        print("  - Fig V-2: *_zoom_alignment.png (局部对齐细节)")
        print("  - Fig V-3: *_error_analysis.png (误差统计分析)")
        print("  - Fig V-4: *_deconfounded.png (去混杂相关)")
        print("  - Table V-1: multitrack_validation_summary.csv (多轨汇总)")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()