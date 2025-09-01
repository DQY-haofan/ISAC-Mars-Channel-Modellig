#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
火星SHARAD雷达杂波与MOLA地形数据对齐验证脚本
用途：验证SHARAD官方杂波图与MOLA高程数据的几何一致性
作者：火星通信专家
"""

import os
import numpy as np
import pvl
import pandas as pd
import matplotlib.pyplot as plt
from pds4_tools import pds4_read
from matplotlib.gridspec import GridSpec

# ===========================
# 配置数据路径
# ===========================
# MOLA MEGDR 数据路径
MOLA_DIR = "MOLA MEGDR"
MOLA_IMG = os.path.join(MOLA_DIR, "megt00n090hb.img")
MOLA_LBL = os.path.join(MOLA_DIR, "megt00n090hb.lbl")

# SHARAD Clutter 数据路径
SHARAD_DIR = "SHARAD Clutter"
SIM_IMG = os.path.join(SHARAD_DIR, "s_00810101_sim.img")
SIM_XML = os.path.join(SHARAD_DIR, "s_00810101_sim.xml")
EMAP_IMG = os.path.join(SHARAD_DIR, "s_00810101_emap.img")
EMAP_XML = os.path.join(SHARAD_DIR, "s_00810101_emap.xml")
RTRN_CSV = os.path.join(SHARAD_DIR, "s_00810101_rtrn.csv")
RTRN_XML = os.path.join(SHARAD_DIR, "s_00810101_rtrn.xml")


# ===========================
# MOLA MEGDR 读取函数
# ===========================
def read_megdr(img_path, lbl_path):
    """
    读取MOLA MEGDR瓦片数据

    参数:
        img_path: .img文件路径
        lbl_path: .lbl标签文件路径

    返回:
        arr: 高程数组（米）
        meta: 元数据字典（包含地理范围和分辨率）
    """
    print(f"读取MOLA MEGDR数据: {img_path}")

    # 解析LBL标签文件
    lbl = pvl.load(lbl_path)
    lines = int(lbl['IMAGE']['LINES'])
    cols = int(lbl['IMAGE']['LINE_SAMPLES'])
    sbits = int(lbl['IMAGE']['SAMPLE_BITS'])
    stype = str(lbl['IMAGE']['SAMPLE_TYPE'])
    scale = float(lbl['IMAGE'].get('SCALING_FACTOR', 1))
    offset = float(lbl['IMAGE'].get('OFFSET', 0))

    # 确定数据类型
    if sbits == 16 and 'MSB' in stype:
        dtype = '>i2'  # 大端16位整数
    elif sbits == 16 and 'LSB' in stype:
        dtype = '<i2'  # 小端16位整数
    else:
        raise ValueError(f'未支持的数据类型: {sbits}位 {stype}')

    # 读取二进制数据
    arr = np.fromfile(img_path, dtype=dtype).reshape(lines, cols)
    arr = arr * scale + offset  # 应用缩放和偏移

    # 提取地理投影信息
    proj = lbl['IMAGE_MAP_PROJECTION']
    meta = dict(
        west=float(proj['WESTERNMOST_LONGITUDE']),
        east=float(proj['EASTERNMOST_LONGITUDE']),
        lat_max=float(proj['MAXIMUM_LATITUDE']),
        lat_min=float(proj['MINIMUM_LATITUDE']),
        ppd=float(proj['MAP_RESOLUTION']),  # 每度像素数（通常为128）
        lines=lines,
        cols=cols
    )

    print(f"  瓦片覆盖范围: 经度 {meta['west']}°E - {meta['east']}°E, "
          f"纬度 {meta['lat_min']}°N - {meta['lat_max']}°N")
    print(f"  分辨率: {meta['ppd']} 像素/度")
    print(f"  数组尺寸: {lines} × {cols}")

    return arr, meta


def sample_by_latlon(arr, meta, lonE, lat):
    """
    根据经纬度从MOLA数组中采样高程值

    参数:
        arr: MOLA高程数组
        meta: 元数据字典
        lonE: 经度（0-360°E）
        lat: 纬度（-90到90°N）

    返回:
        高程值（米），超出范围返回NaN
    """
    lonE = lonE % 360.0  # 确保经度在0-360范围内

    # 检查是否在瓦片覆盖范围内
    if not (meta['west'] <= lonE <= meta['east'] and
            meta['lat_min'] <= lat <= meta['lat_max']):
        return np.nan

    # 计算像素坐标
    j = int((lonE - meta['west']) * meta['ppd'])  # 列索引
    i = int((meta['lat_max'] - lat) * meta['ppd'])  # 行索引（从北往南）

    # 边界检查
    if i < 0 or i >= arr.shape[0] or j < 0 or j >= arr.shape[1]:
        return np.nan

    return float(arr[i, j])


# ===========================
# SHARAD Clutter 读取函数
# ===========================
def read_sharad_clutter(sim_xml, emap_xml=None, rtrn_csv=None):
    """
    读取SHARAD Clutter套件数据

    参数:
        sim_xml: 杂波仿真图XML路径
        emap_xml: 能量映射图XML路径（可选）
        rtrn_csv: 几何返回数据CSV路径（可选）

    返回:
        字典包含sim、emap、rtrn数据
    """
    result = {}

    # 读取杂波仿真图
    print(f"读取SHARAD杂波仿真数据: {sim_xml}")
    prod = pds4_read(sim_xml)
    result['sim'] = prod[0].data  # 2D杂波功率（延迟×沿轨）
    print(f"  杂波图尺寸: {result['sim'].shape}")

    # 读取能量映射图（如果提供）
    if emap_xml and os.path.exists(emap_xml):
        print(f"读取SHARAD能量映射数据: {emap_xml}")
        prod = pds4_read(emap_xml)
        result['emap'] = prod[0].data
        print(f"  能量图尺寸: {result['emap'].shape}")

    # 读取几何数据（如果提供）
    if rtrn_csv and os.path.exists(rtrn_csv):
        print(f"读取SHARAD几何数据: {rtrn_csv}")
        result['rtrn'] = pd.read_csv(rtrn_csv)
        print(f"  几何数据行数: {len(result['rtrn'])}")
        print(f"  可用列: {', '.join(result['rtrn'].columns[:10])}...")

    return result


# ===========================
# 主分析函数
# ===========================
def analyze_sharad_mola_alignment():
    """
    执行SHARAD杂波与MOLA地形的对齐分析
    """
    print("=" * 60)
    print("火星SHARAD雷达杂波与MOLA地形对齐验证")
    print("=" * 60)

    # 1. 读取MOLA数据
    print("\n[步骤1] 读取MOLA MEGDR高程数据")
    dem, meta = read_megdr(MOLA_IMG, MOLA_LBL)

    # 2. 读取SHARAD数据
    print("\n[步骤2] 读取SHARAD Clutter数据")
    sharad_data = read_sharad_clutter(SIM_XML, EMAP_XML, RTRN_CSV)

    # 3. 提取几何信息
    print("\n[步骤3] 提取轨道几何信息")
    if 'rtrn' not in sharad_data:
        print("警告：未找到RTRN几何数据，跳过对齐分析")
        return

    df = sharad_data['rtrn']

    # 自动查找经纬度列名（不同版本可能略有差异）
    cols = {c.lower(): c for c in df.columns}
    lon_col = cols.get('longitude') or cols.get('lon') or cols.get('east_longitude')
    lat_col = cols.get('latitude') or cols.get('lat')

    if lon_col is None or lat_col is None:
        print("错误：无法在RTRN数据中找到经纬度列")
        print(f"可用列名: {list(df.columns)}")
        return

    print(f"  使用经度列: {lon_col}")
    print(f"  使用纬度列: {lat_col}")

    # 4. 沿轨采样MOLA高程
    print("\n[步骤4] 沿轨道采样MOLA高程")
    track_elev = []
    valid_count = 0

    for idx, (lonE, lat) in enumerate(zip(df[lon_col].values, df[lat_col].values)):
        elev = sample_by_latlon(dem, meta, float(lonE), float(lat))
        track_elev.append(elev)
        if not np.isnan(elev):
            valid_count += 1

        # 每100个点打印一次进度
        if (idx + 1) % 100 == 0:
            print(f"  处理进度: {idx + 1}/{len(df)} 点")

    track_elev = np.array(track_elev)
    print(f"  完成！有效高程点: {valid_count}/{len(track_elev)}")

    # 5. 计算相关性分析
    print("\n[步骤5] 相关性分析")
    sim = sharad_data['sim']

    # 选择近地表延迟窗（例如20-40 bins）
    surface_window = slice(20, 40)
    surface_power = np.nanmax(sim[surface_window, :], axis=0)

    # 确保长度匹配
    min_len = min(len(surface_power), len(track_elev))
    surface_power = surface_power[:min_len]
    track_elev_subset = track_elev[:min_len]

    # 计算相关系数（忽略NaN值）
    valid_mask = ~np.isnan(track_elev_subset)
    if np.sum(valid_mask) > 10:
        corr = np.corrcoef(surface_power[valid_mask],
                           track_elev_subset[valid_mask])[0, 1]
        print(f"  近地表回波强度与地形高程相关系数: {corr:.3f}")

    # 6. 可视化结果
    print("\n[步骤6] 生成可视化图表")
    create_comprehensive_plot(sim, track_elev, sharad_data.get('emap'),
                              df, lon_col, lat_col)

    print("\n分析完成！")
    print("=" * 60)


def create_comprehensive_plot(sim, track_elev, emap, df, lon_col, lat_col):
    """
    创建综合可视化图表
    """
    # 创建图形布局
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 0.8])

    # 1. 杂波仿真图
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(sim, aspect='auto', cmap='gray',
                     vmin=np.percentile(sim, 5),
                     vmax=np.percentile(sim, 95))
    ax1.set_title("SHARAD Surface Clutter Simulation (sim)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Delay bin")
    ax1.set_xlabel("Along-track column")
    plt.colorbar(im1, ax=ax1, label="Power (dB)")

    # 2. 能量映射图（如果有）
    if emap is not None:
        ax2 = fig.add_subplot(gs[1, :])
        im2 = ax2.imshow(emap, aspect='auto', cmap='hot',
                         vmin=np.percentile(emap, 5),
                         vmax=np.percentile(emap, 95))
        ax2.set_title("SHARAD Energy Map (emap)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Delay bin")
        ax2.set_xlabel("Along-track column")
        plt.colorbar(im2, ax=ax2, label="Energy")

    # 3. 沿轨高程曲线
    ax3 = fig.add_subplot(gs[2, :])
    valid_mask = ~np.isnan(track_elev)
    ax3.plot(track_elev, 'b-', linewidth=1.5, alpha=0.7, label='MOLA Elevation')
    ax3.fill_between(range(len(track_elev)),
                     np.nanmin(track_elev), track_elev,
                     where=valid_mask, alpha=0.3, color='blue')
    ax3.set_title("Along-track MOLA Elevation Profile", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Elevation (m)")
    ax3.set_xlabel("Along-track index")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 显示统计信息
    stats_text = f"Min: {np.nanmin(track_elev):.0f} m\n"
    stats_text += f"Max: {np.nanmax(track_elev):.0f} m\n"
    stats_text += f"Range: {np.nanmax(track_elev) - np.nanmin(track_elev):.0f} m"
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. 轨道地图
    ax4 = fig.add_subplot(gs[3, 0])
    sc = ax4.scatter(df[lon_col], df[lat_col],
                     c=track_elev[:len(df)],
                     s=1, cmap='terrain', alpha=0.8)
    ax4.set_title("Ground Track", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Longitude (°E)")
    ax4.set_ylabel("Latitude (°N)")
    ax4.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax4, label="Elevation (m)")

    # 5. 高程分布直方图
    ax5 = fig.add_subplot(gs[3, 1])
    valid_elevs = track_elev[~np.isnan(track_elev)]
    ax5.hist(valid_elevs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax5.set_title("Elevation Distribution", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Elevation (m)")
    ax5.set_ylabel("Count")
    ax5.grid(True, alpha=0.3)

    # 添加统计线
    ax5.axvline(np.mean(valid_elevs), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(valid_elevs):.0f} m')
    ax5.axvline(np.median(valid_elevs), color='green', linestyle='--',
                linewidth=2, label=f'Median: {np.median(valid_elevs):.0f} m')
    ax5.legend()

    plt.suptitle("SHARAD-MOLA Alignment Verification Analysis",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # 保存图像
    output_file = "sharad_mola_alignment_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  图表已保存到: {output_file}")

    plt.show()


# ===========================
# 额外分析功能
# ===========================
def extract_surface_echo_profile(sim, delay_range=(20, 40)):
    """
    从杂波图中提取地表回波剖面

    参数:
        sim: 杂波仿真数组
        delay_range: 地表回波的延迟范围

    返回:
        地表回波强度剖面
    """
    surface_echo = np.nanmax(sim[delay_range[0]:delay_range[1], :], axis=0)
    return surface_echo


def compute_roughness_proxy(track_elev, window_size=10):
    """
    计算地形粗糙度代理（局部标准差）

    参数:
        track_elev: 沿轨高程数组
        window_size: 滑动窗口大小

    返回:
        粗糙度数组
    """
    roughness = np.zeros_like(track_elev)
    for i in range(len(track_elev)):
        start = max(0, i - window_size // 2)
        end = min(len(track_elev), i + window_size // 2 + 1)
        window = track_elev[start:end]
        valid = window[~np.isnan(window)]
        if len(valid) > 1:
            roughness[i] = np.std(valid)
        else:
            roughness[i] = np.nan
    return roughness


# ===========================
# 主程序入口
# ===========================
if __name__ == "__main__":
    try:
        # 检查必要的文件是否存在
        required_files = [MOLA_IMG, MOLA_LBL, SIM_XML]
        missing_files = [f for f in required_files if not os.path.exists(f)]

        if missing_files:
            print("错误：以下必需文件缺失:")
            for f in missing_files:
                print(f"  - {f}")
            print("\n请确保文件放在正确的文件夹中:")
            print("  - MOLA MEGDR/")
            print("  - SHARAD Clutter/")
            exit(1)

        # 执行主分析
        analyze_sharad_mola_alignment()

        # 可选：执行额外分析
        print("\n是否执行额外粗糙度分析？(y/n): ", end="")
        if input().lower() == 'y':
            print("\n执行地形粗糙度分析...")
            # 这里可以添加更多分析代码

    except Exception as e:
        print(f"\n错误发生: {e}")
        import traceback

        traceback.print_exc()