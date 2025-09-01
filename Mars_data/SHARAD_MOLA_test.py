#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级版 - 火星SHARAD雷达杂波与MOLA地形定量对齐验证
改进内容：
1. 多瓦片无缝采样（支持完整轨道）
2. 使用NadirLon/Lat（星下点）
3. 叠加理论表面回波线
4. Spearman相关性分析
5. 稳健的粗糙度计算
"""

import os
import numpy as np
import pvl
import pandas as pd
import matplotlib.pyplot as plt
from pds4_tools import pds4_read
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===========================
# 配置数据路径
# ===========================
MOLA_DIR = "MOLA MEGDR"
SHARAD_DIR = "SHARAD Clutter"

# MOLA瓦片（两张拼接实现完整覆盖）
MOLA_N_IMG = os.path.join(MOLA_DIR, "megt00n090hb.img")  # -44° to 0°N
MOLA_N_LBL = os.path.join(MOLA_DIR, "megt00n090hb.lbl")
MOLA_S_IMG = os.path.join(MOLA_DIR, "megt44s090hb.img")  # -88° to -44°N
MOLA_S_LBL = os.path.join(MOLA_DIR, "megt44s090hb.lbl")

# SHARAD数据
SIM_XML = os.path.join(SHARAD_DIR, "s_00810101_sim.xml")
EMAP_XML = os.path.join(SHARAD_DIR, "s_00810101_emap.xml")
RTRN_CSV = os.path.join(SHARAD_DIR, "s_00810101_rtrn.csv")

# ===========================
# 全局变量（存储读取的DEM）
# ===========================
DEM_N, META_N = None, None  # 北瓦片
DEM_S, META_S = None, None  # 南瓦片


# ===========================
# MOLA读取和处理函数
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

    print(f"    范围: {meta['west']}°-{meta['east']}°E, "
          f"{meta['lat_min']}°-{meta['lat_max']}°N")

    return arr, meta


def bilinear_sample(dem, meta, lonE, lat):
    """双线性插值采样"""
    # 转换为像素坐标（浮点）
    jf = (lonE - meta['west']) * meta['ppd']
    if_val = (meta['lat_max'] - lat) * meta['ppd']

    # 边界检查
    if not (0 <= jf < meta['cols'] - 1) or not (0 <= if_val < meta['lines'] - 1):
        return np.nan

    # 整数索引和分数部分
    i0, j0 = int(if_val), int(jf)
    di, dj = if_val - i0, jf - j0

    # 双线性插值
    v00 = dem[i0, j0]
    v01 = dem[i0, j0 + 1]
    v10 = dem[i0 + 1, j0]
    v11 = dem[i0 + 1, j0 + 1]

    return (1 - di) * (1 - dj) * v00 + (1 - di) * dj * v01 + di * (1 - dj) * v10 + di * dj * v11


def sample_mola(lonE, lat):
    """智能选择瓦片并采样"""
    lonE = lonE % 360.0

    # 检查南瓦片（-88° to -44°）
    if (META_S is not None and
            META_S['west'] <= lonE <= META_S['east'] and
            META_S['lat_min'] <= lat <= META_S['lat_max']):
        return bilinear_sample(DEM_S, META_S, lonE, lat)

    # 检查北瓦片（-44° to 0°）
    if (META_N is not None and
            META_N['west'] <= lonE <= META_N['east'] and
            META_N['lat_min'] <= lat <= META_N['lat_max']):
        return bilinear_sample(DEM_N, META_N, lonE, lat)

    return np.nan


# ===========================
# SHARAD读取函数
# ===========================
def read_sharad_clutter(sim_xml, emap_xml=None, rtrn_csv=None):
    """读取SHARAD Clutter套件数据"""
    result = {}

    print(f"读取SHARAD杂波数据...")
    prod = pds4_read(sim_xml)

    if len(prod) > 1:
        print(f"  发现{len(prod)}个数组，使用组合杂波（最后一个）")
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
        print(f"  可用列: {list(result['rtrn'].columns)}")

    return result


# ===========================
# 主分析函数（高级版）
# ===========================
def advanced_sharad_mola_analysis():
    """执行高级SHARAD-MOLA定量对齐分析"""

    print("=" * 70)
    print("高级SHARAD-MOLA定量对齐验证")
    print("=" * 70)

    # 1. 读取多个MOLA瓦片
    print("\n[步骤1] 读取MOLA MEGDR瓦片（多瓦片拼接）")
    global DEM_N, META_N, DEM_S, META_S

    if os.path.exists(MOLA_N_IMG):
        DEM_N, META_N = read_megdr(MOLA_N_IMG, MOLA_N_LBL)
    else:
        print(f"  警告：北瓦片不存在")

    if os.path.exists(MOLA_S_IMG):
        DEM_S, META_S = read_megdr(MOLA_S_IMG, MOLA_S_LBL)
    else:
        print(f"  警告：南瓦片不存在")

    # 2. 读取SHARAD数据
    print("\n[步骤2] 读取SHARAD Clutter数据")
    sharad_data = read_sharad_clutter(SIM_XML, EMAP_XML, RTRN_CSV)

    if 'rtrn' not in sharad_data:
        print("错误：无RTRN数据")
        return

    df = sharad_data['rtrn']

    # 3. 智能选择经纬度列（优先使用NadirLon/Lat）
    print("\n[步骤3] 选择最优经纬度源")

    # 优先级列表
    lon_priority = ['NadirLon', 'FirstLon', 'SpacecraftLon']
    lat_priority = ['NadirLat', 'FirstLat', 'SpacecraftLat']

    lon_col = None
    lat_col = None

    # 查找可用的经纬度列
    available_cols = df.columns.tolist()

    for col in lon_priority:
        if col in available_cols:
            lon_col = col
            break

    for col in lat_priority:
        if col in available_cols:
            lat_col = col
            break

    # 如果还是找不到，尝试模糊匹配
    if not lon_col:
        for col in available_cols:
            if 'lon' in col.lower():
                lon_col = col
                break

    if not lat_col:
        for col in available_cols:
            if 'lat' in col.lower():
                lat_col = col
                break

    if not lon_col or not lat_col:
        print("错误：无法找到经纬度列")
        return

    print(f"  使用经度源: {lon_col}")
    print(f"  使用纬度源: {lat_col}")

    # 显示覆盖范围
    lon_range = (df[lon_col].min(), df[lon_col].max())
    lat_range = (df[lat_col].min(), df[lat_col].max())
    print(f"  轨道覆盖: {lon_range[0]:.2f}°-{lon_range[1]:.2f}°E, "
          f"{lat_range[0]:.2f}°-{lat_range[1]:.2f}°N")

    # 4. 沿轨采样（使用多瓦片）
    print("\n[步骤4] 沿轨采样MOLA高程（多瓦片无缝采样）")
    track_elev = []
    valid_count = 0

    for idx, (lonE, lat) in enumerate(zip(df[lon_col].values, df[lat_col].values)):
        # 处理负经度
        if lonE < 0:
            lonE += 360

        elev = sample_mola(float(lonE), float(lat))
        track_elev.append(elev)

        if not np.isnan(elev):
            valid_count += 1

        if (idx + 1) % 1000 == 0:
            pct = valid_count / (idx + 1) * 100
            print(f"  进度: {idx + 1}/{len(df)}, 有效率: {pct:.1f}%")

    track_elev = np.array(track_elev)
    final_pct = valid_count / len(track_elev) * 100
    print(f"  完成！有效点: {valid_count}/{len(track_elev)} ({final_pct:.1f}%)")

    if valid_count < 100:
        print("警告：有效点太少，无法进行有意义的分析")
        return

    # 5. 稳健的相关性分析
    print("\n[步骤5] 定量相关性分析")
    sim = sharad_data['sim']

    # 提取近地表功率
    surface_window = slice(20, min(40, sim.shape[0]))
    surface_power = np.nanmax(sim[surface_window, :], axis=0)

    # 确保长度匹配
    min_len = min(len(surface_power), len(track_elev))
    surface_power = surface_power[:min_len]
    track_elev_subset = track_elev[:min_len]

    # 计算Spearman相关（更稳健）
    valid_mask = ~np.isnan(track_elev_subset) & np.isfinite(surface_power)
    valid_count_corr = np.sum(valid_mask)

    if valid_count_corr > 100:
        rho, p_value = spearmanr(track_elev_subset[valid_mask],
                                 surface_power[valid_mask])
        print(f"  Spearman相关系数: ρ = {rho:.3f} (p = {p_value:.2e})")

        # 也计算Pearson相关作为对比
        try:
            pearson_corr = np.corrcoef(track_elev_subset[valid_mask],
                                       surface_power[valid_mask])[0, 1]
            print(f"  Pearson相关系数: r = {pearson_corr:.3f}")
        except:
            print(f"  Pearson相关计算失败")
    else:
        print(f"  有效点不足({valid_count_corr})，无法计算相关性")

    # 统计信息
    valid_elevs = track_elev[~np.isnan(track_elev)]
    if len(valid_elevs) > 0:
        print(f"\n  高程统计:")
        print(f"    范围: {np.min(valid_elevs):.0f} - {np.max(valid_elevs):.0f} m")
        print(f"    均值: {np.mean(valid_elevs):.0f} m")
        print(f"    中位数: {np.median(valid_elevs):.0f} m")
        print(f"    标准差: {np.std(valid_elevs):.0f} m")

    # 6. 可视化（包含理论表面线）
    print("\n[步骤6] 生成高级可视化")
    create_advanced_visualization(sharad_data, track_elev, df, lon_col, lat_col)

    # 7. 稳健的粗糙度分析
    print("\n[步骤7] 稳健粗糙度分析")
    compute_robust_roughness(track_elev)

    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)

    return sharad_data, track_elev, df


def create_advanced_visualization(sharad_data, track_elev, df, lon_col, lat_col):
    """创建高级可视化（包含理论表面线）"""

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.2, 0.6, 1, 0.8],
                  hspace=0.25, wspace=0.25)

    sim = sharad_data['sim']

    # 1. 杂波图 + 理论表面线
    ax1 = fig.add_subplot(gs[0, :])

    # 对数尺度显示
    sim_display = np.log10(np.abs(sim) + 1e-10)
    im1 = ax1.imshow(sim_display, aspect='auto', cmap='gray',
                     vmin=np.percentile(sim_display, 5),
                     vmax=np.percentile(sim_display, 95))

    # 叠加理论表面线（如果有NadirLine）
    if 'NadirLine' in df.columns:
        ax1.plot(np.arange(len(df)), df['NadirLine'].values,
                 'c-', linewidth=1.5, alpha=0.8, label='Predicted Surface')
        ax1.legend(loc='upper right')
        print("  ✓ 已叠加理论表面回波线")
    elif 'SpacecraftHgt' in df.columns and not np.all(np.isnan(track_elev)):
        # 尝试根据高程差估算延迟
        try:
            # 简化计算：假设3.75 μs/bin（SHARAD标准）
            c = 3e8  # 光速
            dt = 0.375e-6  # 每bin时间间隔

            spacecraft_hgt = df['SpacecraftHgt'].values[:len(track_elev)]
            valid_mask = ~np.isnan(track_elev)

            # 计算双程延迟
            delay_bins = np.zeros(len(track_elev))
            delay_bins[valid_mask] = 2 * (spacecraft_hgt[valid_mask] -
                                          track_elev[valid_mask]) / (c * dt)

            ax1.plot(np.arange(len(delay_bins)), delay_bins,
                     'y--', linewidth=1, alpha=0.6, label='Estimated Surface')
            ax1.legend(loc='upper right')
            print("  ✓ 基于高程差估算表面线")
        except:
            print("  ⚠ 无法计算理论表面线")

    ax1.set_title("SHARAD Clutter with Surface Echo Prediction",
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel("Delay bin")
    ax1.set_xlabel("Along-track column")
    plt.colorbar(im1, ax=ax1, label="Log10(Power)")

    # 2. 能量图
    if 'emap' in sharad_data:
        ax2 = fig.add_subplot(gs[1, :])
        emap = sharad_data['emap']
        im2 = ax2.imshow(emap, aspect='auto', cmap='hot',
                         vmin=np.percentile(emap, 5),
                         vmax=np.percentile(emap, 95))
        ax2.set_title("Energy Map", fontsize=11)
        ax2.set_ylabel("Delay bin")
        ax2.set_xlabel("Along-track column")
        plt.colorbar(im2, ax=ax2, label="Energy")

    # 3. 沿轨高程（显示有效/无效段）
    ax3 = fig.add_subplot(gs[2, :])

    valid_mask = ~np.isnan(track_elev)
    x = np.arange(len(track_elev))

    # 分段显示
    ax3.plot(x[valid_mask], track_elev[valid_mask], 'b-',
             linewidth=1.5, alpha=0.8, label='Valid MOLA')

    # 标记无效段
    invalid_segments = []
    in_invalid = False
    start_idx = 0

    for i in range(len(valid_mask)):
        if not valid_mask[i] and not in_invalid:
            start_idx = i
            in_invalid = True
        elif valid_mask[i] and in_invalid:
            invalid_segments.append((start_idx, i - 1))
            in_invalid = False

    if in_invalid:
        invalid_segments.append((start_idx, len(valid_mask) - 1))

    for start, end in invalid_segments[:5]:  # 只标记前5个
        ax3.axvspan(start, end, alpha=0.2, color='red')

    ax3.set_title("Along-track MOLA Elevation (Multi-tile Sampling)",
                  fontsize=12, fontweight='bold')
    ax3.set_ylabel("Elevation (m)")
    ax3.set_xlabel("Along-track index")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 添加统计框
    valid_elevs = track_elev[valid_mask]
    if len(valid_elevs) > 0:
        stats_text = (f"Coverage: {np.sum(valid_mask) / len(track_elev) * 100:.1f}%\n"
                      f"Range: {np.min(valid_elevs):.0f} - {np.max(valid_elevs):.0f} m\n"
                      f"Mean: {np.mean(valid_elevs):.0f} m")
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 4. 轨道地图（彩色编码）
    ax4 = fig.add_subplot(gs[3, 0])

    lons = df[lon_col].values.copy()
    lons[lons < 0] += 360

    sc = ax4.scatter(lons, df[lat_col], c=track_elev[:len(df)],
                     s=2, cmap='terrain', alpha=0.8)
    ax4.set_title("Ground Track Coverage", fontsize=11)
    ax4.set_xlabel("Longitude (°E)")
    ax4.set_ylabel("Latitude (°N)")
    ax4.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax4, label="Elevation (m)")

    # 标记瓦片边界
    if META_N and META_S:
        ax4.axhline(y=-44, color='red', linestyle='--',
                    linewidth=1, alpha=0.5, label='Tile boundary')
        ax4.legend(loc='best', fontsize=8)

    # 5. 相关性散点图
    ax5 = fig.add_subplot(gs[3, 1])

    # 近地表功率 vs 高程
    surface_power = np.nanmax(sim[20:40, :], axis=0)
    min_len = min(len(surface_power), len(track_elev))

    valid = ~np.isnan(track_elev[:min_len]) & np.isfinite(surface_power[:min_len])

    if np.sum(valid) > 10:
        ax5.scatter(track_elev[:min_len][valid], surface_power[:min_len][valid],
                    s=1, alpha=0.5, color='blue')
        ax5.set_xlabel("MOLA Elevation (m)")
        ax5.set_ylabel("Surface Echo Power")
        ax5.set_title("Elevation vs Echo Power", fontsize=11)
        ax5.grid(True, alpha=0.3)

        # 添加趋势线
        try:
            z = np.polyfit(track_elev[:min_len][valid],
                           surface_power[:min_len][valid], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(np.min(track_elev[:min_len][valid]),
                                  np.max(track_elev[:min_len][valid]), 100)
            ax5.plot(x_trend, p(x_trend), "r-", alpha=0.5, linewidth=2)
        except:
            pass

    plt.suptitle("Advanced SHARAD-MOLA Quantitative Alignment Analysis",
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    output_file = "advanced_sharad_mola_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {output_file}")
    plt.show()


def compute_robust_roughness(track_elev, window_size=50):
    """计算稳健的地形粗糙度"""

    # 插值填充NaN（而不是用均值）
    valid_mask = ~np.isnan(track_elev)

    if np.sum(valid_mask) < window_size * 2:
        print("  数据不足，无法计算粗糙度")
        return

    # 创建插值函数
    valid_indices = np.where(valid_mask)[0]
    valid_values = track_elev[valid_mask]

    # 线性插值
    track_elev_filled = track_elev.copy()
    invalid_indices = np.where(~valid_mask)[0]

    if len(invalid_indices) > 0 and len(valid_indices) > 1:
        # 使用线性插值填充
        f = interp1d(valid_indices, valid_values,
                     kind='linear', fill_value='extrapolate')
        track_elev_filled[invalid_indices] = f(invalid_indices)

    # 计算局部标准差作为粗糙度
    roughness = np.zeros_like(track_elev_filled)
    half_window = window_size // 2

    for i in range(len(track_elev_filled)):
        start = max(0, i - half_window)
        end = min(len(track_elev_filled), i + half_window + 1)

        window_data = track_elev_filled[start:end]
        if len(window_data) > 1:
            roughness[i] = np.std(window_data)

    # 只在有效段计算统计
    roughness_valid = roughness[valid_mask]

    # 可视化
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 原始高程
    ax1.plot(track_elev, 'b-', alpha=0.7, label='Original')
    ax1.plot(track_elev_filled, 'r--', alpha=0.5, linewidth=0.5,
             label='Interpolated')
    ax1.set_ylabel('Elevation (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Elevation Profile')

    # 粗糙度
    ax2.plot(roughness, 'g-', alpha=0.7)
    ax2.set_ylabel('Roughness (Std Dev, m)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Local Roughness (window={window_size} samples)')

    # 标记高粗糙度区域
    threshold = np.percentile(roughness_valid, 90)
    high_rough = roughness > threshold
    ax2.fill_between(range(len(roughness)), 0, roughness,
                     where=high_rough, alpha=0.3, color='red',
                     label=f'High roughness (>{threshold:.0f}m)')
    ax2.legend()

    # 坡度（一阶差分）
    slope = np.gradient(track_elev_filled)
    ax3.plot(np.abs(slope), 'orange', alpha=0.7)
    ax3.set_ylabel('|Slope| (m/sample)')
    ax3.set_xlabel('Along-track index')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Absolute Slope')

    plt.suptitle('Robust Terrain Roughness Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('robust_roughness_analysis.png', dpi=150)
    plt.show()

    # 输出统计
    print(f"\n  粗糙度统计:")
    print(f"    平均: {np.mean(roughness_valid):.1f} m")
    print(f"    中位数: {np.median(roughness_valid):.1f} m")
    print(f"    90百分位: {threshold:.1f} m")
    print(f"    最大值: {np.max(roughness_valid):.1f} m")

    # 识别主要粗糙区域
    rough_regions = []
    in_rough = False
    start_idx = 0

    for i in range(len(high_rough)):
        if high_rough[i] and not in_rough:
            start_idx = i
            in_rough = True
        elif not high_rough[i] and in_rough:
            rough_regions.append((start_idx, i - 1))
            in_rough = False

    if rough_regions:
        print(f"    主要粗糙区域 (前5个):")
        for start, end in rough_regions[:5]:
            print(f"      索引 {start}-{end}: "
                  f"粗糙度 {np.mean(roughness[start:end + 1]):.0f} m")


# ===========================
# 主程序
# ===========================
if __name__ == "__main__":
    try:
        # 检查文件
        required = [SIM_XML, RTRN_CSV]
        missing = [f for f in required if not os.path.exists(f)]

        if missing:
            print("错误：缺少必需文件:")
            for f in missing:
                print(f"  - {f}")
            exit(1)

        # 执行高级分析
        results = advanced_sharad_mola_analysis()

        if results:
            print("\n✓ 所有分析完成！")
            print("\n生成的文件:")
            print("  - advanced_sharad_mola_analysis.png")
            print("  - robust_roughness_analysis.png")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()