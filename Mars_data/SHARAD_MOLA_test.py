#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
火星SHARAD雷达杂波与MOLA地形数据对齐验证脚本 - 修复版
修复：正确识别RTRN数据中的经纬度列名
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

    注意：PDS4数据可能包含多个数组（左侧、右侧、组合杂波）
    """
    result = {}

    # 读取杂波仿真图
    print(f"读取SHARAD杂波仿真数据: {sim_xml}")
    prod = pds4_read(sim_xml)

    # 处理可能的多个数组
    if len(prod) > 1:
        print(f"  发现{len(prod)}个数组:")
        for i, arr in enumerate(prod):
            try:
                # 尝试不同的方式获取数组信息
                if hasattr(arr, 'meta') and hasattr(arr.meta, 'get'):
                    label = arr.meta.get('local_identifier', f'Array_{i}')
                elif hasattr(arr, 'label'):
                    label = arr.label if isinstance(arr.label, str) else f'Array_{i}'
                elif hasattr(arr, 'id'):
                    label = arr.id
                else:
                    label = f'Array_{i}'

                # 获取数据形状
                if hasattr(arr, 'data'):
                    shape = arr.data.shape
                elif hasattr(arr, 'shape'):
                    shape = arr.shape
                else:
                    shape = 'Unknown shape'

                print(f"    [{i}] {label}: {shape}")
            except Exception as e:
                print(f"    [{i}] Array_{i}: Unable to get info ({e})")

        # 通常使用组合杂波（Combined_Clutter_Simulation）
        # 它通常是最后一个数组
        try:
            result['sim'] = prod[-1].data if hasattr(prod[-1], 'data') else prod[-1]
            print(f"  使用组合杂波图 (数组[{len(prod) - 1}]): {result['sim'].shape}")
        except Exception as e:
            print(f"  警告：无法获取数组数据，尝试使用第一个数组")
            result['sim'] = prod[0].data if hasattr(prod[0], 'data') else prod[0]
            print(f"  使用数组[0]: {result['sim'].shape}")
    else:
        result['sim'] = prod[0].data if hasattr(prod[0], 'data') else prod[0]
        print(f"  杂波图尺寸: {result['sim'].shape}")

    # 读取能量映射图（如果提供）
    if emap_xml and os.path.exists(emap_xml):
        print(f"读取SHARAD能量映射数据: {emap_xml}")
        try:
            prod = pds4_read(emap_xml)
            result['emap'] = prod[0].data if hasattr(prod[0], 'data') else prod[0]
            print(f"  能量图尺寸: {result['emap'].shape}")
        except Exception as e:
            print(f"  警告：无法读取能量映射图: {e}")

    # 读取几何数据（如果提供）
    if rtrn_csv and os.path.exists(rtrn_csv):
        print(f"读取SHARAD几何数据: {rtrn_csv}")
        result['rtrn'] = pd.read_csv(rtrn_csv)
        print(f"  几何数据行数: {len(result['rtrn'])}")
        print(f"  可用列: {', '.join(result['rtrn'].columns[:10])}...")

    return result


# ===========================
# 主分析函数 - 修复版
# ===========================
def analyze_sharad_mola_alignment():
    """
    执行SHARAD杂波与MOLA地形的对齐分析
    修复：正确处理不同的列名格式
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

    # 打印所有列名以便调试
    print(f"  RTRN数据包含 {len(df.columns)} 列:")
    for i, col in enumerate(df.columns):
        if i < 20:  # 只打印前20个
            print(f"    {i + 1}. {col}")

    # 智能查找经纬度列名（支持多种命名格式）
    lon_col = None
    lat_col = None

    # 可能的经度列名
    possible_lon_names = ['SpacecraftLon', 'NadirLon', 'FirstLon',
                          'Longitude', 'longitude', 'lon', 'Lon',
                          'LONGITUDE', 'LON', 'east_longitude']

    # 可能的纬度列名
    possible_lat_names = ['SpacecraftLat', 'NadirLat', 'FirstLat',
                          'Latitude', 'latitude', 'lat', 'Lat',
                          'LATITUDE', 'LAT']

    # 查找经度列
    for col_name in possible_lon_names:
        if col_name in df.columns:
            lon_col = col_name
            break

    # 查找纬度列
    for col_name in possible_lat_names:
        if col_name in df.columns:
            lat_col = col_name
            break

    # 如果找不到，让用户选择
    if lon_col is None or lat_col is None:
        print("\n无法自动识别经纬度列，请手动选择：")

        # 寻找包含'lon'的列
        lon_candidates = [col for col in df.columns if 'lon' in col.lower()]
        lat_candidates = [col for col in df.columns if 'lat' in col.lower()]

        print(f"\n可能的经度列: {lon_candidates}")
        print(f"可能的纬度列: {lat_candidates}")

        # 默认使用航天器位置
        if 'SpacecraftLon' in df.columns and 'SpacecraftLat' in df.columns:
            print("\n使用默认选择：SpacecraftLon 和 SpacecraftLat（航天器星下点）")
            lon_col = 'SpacecraftLon'
            lat_col = 'SpacecraftLat'
        else:
            print("\n错误：无法确定经纬度列，请检查数据文件")
            return

    print(f"\n  选定经度列: {lon_col}")
    print(f"  选定纬度列: {lat_col}")

    # 显示经纬度范围
    lon_range = (df[lon_col].min(), df[lon_col].max())
    lat_range = (df[lat_col].min(), df[lat_col].max())
    print(f"  轨道覆盖范围: 经度 {lon_range[0]:.2f}° - {lon_range[1]:.2f}°")
    print(f"               纬度 {lat_range[0]:.2f}° - {lat_range[1]:.2f}°")

    # 4. 沿轨采样MOLA高程
    print("\n[步骤4] 沿轨道采样MOLA高程")
    track_elev = []
    valid_count = 0
    out_of_range = 0

    for idx, (lonE, lat) in enumerate(zip(df[lon_col].values, df[lat_col].values)):
        # 处理可能的负经度值
        if lonE < 0:
            lonE = lonE + 360

        elev = sample_by_latlon(dem, meta, float(lonE), float(lat))
        track_elev.append(elev)

        if not np.isnan(elev):
            valid_count += 1
        else:
            out_of_range += 1

        # 每500个点打印一次进度
        if (idx + 1) % 500 == 0:
            print(f"  处理进度: {idx + 1}/{len(df)} 点 (有效: {valid_count}, 超范围: {out_of_range})")

    track_elev = np.array(track_elev)
    print(f"  完成！有效高程点: {valid_count}/{len(track_elev)} ({valid_count / len(track_elev) * 100:.1f}%)")

    if valid_count == 0:
        print("\n警告：没有有效的高程数据！可能原因：")
        print("  1. 轨道不在当前MOLA瓦片覆盖范围内")
        print("  2. 经纬度数据格式问题")
        print(f"  当前MOLA瓦片范围: {meta['west']}°-{meta['east']}°E, {meta['lat_min']}°-{meta['lat_max']}°N")
        print(f"  轨道范围: {lon_range[0]:.2f}°-{lon_range[1]:.2f}°E, {lat_range[0]:.2f}°-{lat_range[1]:.2f}°N")
        return

    # 5. 计算相关性分析
    print("\n[步骤5] 相关性分析")
    sim = sharad_data['sim']

    # 选择近地表延迟窗（例如20-40 bins）
    surface_window = slice(20, min(40, sim.shape[0]))
    surface_power = np.nanmax(sim[surface_window, :], axis=0)

    # 确保长度匹配
    min_len = min(len(surface_power), len(track_elev))
    surface_power = surface_power[:min_len]
    track_elev_subset = track_elev[:min_len]

    # 计算相关系数（忽略NaN值）
    valid_mask = ~np.isnan(track_elev_subset)
    if np.sum(valid_mask) > 10:
        # 归一化数据以提高相关性计算的稳定性
        surface_power_norm = (surface_power[valid_mask] - np.mean(surface_power[valid_mask])) / np.std(
            surface_power[valid_mask])
        track_elev_norm = (track_elev_subset[valid_mask] - np.mean(track_elev_subset[valid_mask])) / np.std(
            track_elev_subset[valid_mask])

        corr = np.corrcoef(surface_power_norm, track_elev_norm)[0, 1]
        print(f"  近地表回波强度与地形高程相关系数: {corr:.3f}")

        # 额外统计
        print(f"  高程统计:")
        print(f"    最小值: {np.nanmin(track_elev):.1f} m")
        print(f"    最大值: {np.nanmax(track_elev):.1f} m")
        print(f"    平均值: {np.nanmean(track_elev):.1f} m")
        print(f"    标准差: {np.nanstd(track_elev):.1f} m")

    # 6. 可视化结果
    print("\n[步骤6] 生成可视化图表")
    create_comprehensive_plot(sim, track_elev, sharad_data.get('emap'),
                              df, lon_col, lat_col)

    print("\n分析完成！")
    print("=" * 60)

    return sharad_data, track_elev, df


def create_comprehensive_plot(sim, track_elev, emap, df, lon_col, lat_col):
    """
    创建综合可视化图表
    """
    # 创建图形布局
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 0.5, 1, 0.8], hspace=0.3)

    # 1. 杂波仿真图
    ax1 = fig.add_subplot(gs[0, :])

    # 使用对数尺度显示杂波图以增强对比度
    sim_display = np.log10(np.abs(sim) + 1e-10)  # 避免log(0)

    im1 = ax1.imshow(sim_display, aspect='auto', cmap='gray',
                     vmin=np.percentile(sim_display, 5),
                     vmax=np.percentile(sim_display, 95))
    ax1.set_title("SHARAD Surface Clutter Simulation (log scale)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Delay bin")
    ax1.set_xlabel("Along-track column")
    plt.colorbar(im1, ax=ax1, label="Log10(Power)")

    # 2. 能量映射图（如果有）
    if emap is not None:
        ax2 = fig.add_subplot(gs[1, :])
        im2 = ax2.imshow(emap, aspect='auto', cmap='hot',
                         vmin=np.percentile(emap, 5),
                         vmax=np.percentile(emap, 95))
        ax2.set_title("SHARAD Energy Map", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Delay bin")
        ax2.set_xlabel("Along-track column")
        plt.colorbar(im2, ax=ax2, label="Energy")

    # 3. 沿轨高程曲线
    ax3 = fig.add_subplot(gs[2, :])
    valid_mask = ~np.isnan(track_elev)

    # 主曲线
    ax3.plot(track_elev, 'b-', linewidth=1.5, alpha=0.7, label='MOLA Elevation')

    # 填充区域
    if np.sum(valid_mask) > 0:
        ax3.fill_between(range(len(track_elev)),
                         np.nanmin(track_elev), track_elev,
                         where=valid_mask, alpha=0.3, color='blue')

    ax3.set_title("Along-track MOLA Elevation Profile", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Elevation (m)")
    ax3.set_xlabel("Along-track index")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 显示统计信息
    if np.sum(valid_mask) > 0:
        stats_text = f"Min: {np.nanmin(track_elev):.0f} m\n"
        stats_text += f"Max: {np.nanmax(track_elev):.0f} m\n"
        stats_text += f"Range: {np.nanmax(track_elev) - np.nanmin(track_elev):.0f} m\n"
        stats_text += f"Valid: {np.sum(valid_mask)}/{len(track_elev)}"
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. 轨道地图
    ax4 = fig.add_subplot(gs[3, 0])

    # 处理经度（转换为0-360范围）
    lons = df[lon_col].values.copy()  # 创建副本以避免修改原数据
    lons[lons < 0] += 360

    sc = ax4.scatter(lons, df[lat_col],
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

    if len(valid_elevs) > 0:
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
        results = analyze_sharad_mola_alignment()

        # 可选：执行额外粗糙度分析
        if results:  # 只有在主分析成功时才询问
            user_input = input("\n是否执行额外粗糙度分析？(y/n): ")
            if user_input.lower() == 'y':
                print("\n执行地形粗糙度分析...")
                sharad_data, track_elev, df = results

                # 计算粗糙度
                try:
                    from scipy.ndimage import uniform_filter1d

                    # 使用滑动窗口标准差作为粗糙度指标
                    window_size = 50  # 约50个采样点
                    valid_mask = ~np.isnan(track_elev)

                    if np.sum(valid_mask) > window_size:
                        # 填充NaN值
                        track_elev_filled = np.copy(track_elev)
                        track_elev_filled[~valid_mask] = np.nanmean(track_elev)

                        # 计算局部均值和标准差
                        local_mean = uniform_filter1d(track_elev_filled, window_size)
                        local_var = uniform_filter1d((track_elev_filled - local_mean) ** 2, window_size)
                        roughness = np.sqrt(local_var)

                        # 创建粗糙度图
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

                        ax1.plot(track_elev, 'b-', alpha=0.7, label='Elevation')
                        ax1.set_ylabel('Elevation (m)')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)

                        ax2.plot(roughness, 'r-', alpha=0.7, label='Roughness (Std Dev)')
                        ax2.set_ylabel('Roughness (m)')
                        ax2.set_xlabel('Along-track index')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)

                        plt.suptitle('Terrain Roughness Analysis', fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig('terrain_roughness_analysis.png', dpi=150)
                        plt.show()

                        print("  粗糙度分析完成！")
                        print(f"  平均粗糙度: {np.nanmean(roughness):.1f} m")
                        print(f"  最大粗糙度: {np.nanmax(roughness):.1f} m")
                    else:
                        print("  警告：有效数据点不足，无法计算粗糙度")

                except ImportError:
                    print("  需要安装scipy来执行粗糙度分析: pip install scipy")

    except Exception as e:
        print(f"\n错误发生: {e}")
        import traceback

        traceback.print_exc()