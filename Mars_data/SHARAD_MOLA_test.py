#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整版 - 火星SHARAD雷达三角交叉验证
整合：实测雷达图(Radargram) × 官方杂波(Clutter) × MOLA地形
改进内容：
1. 加入实测radargram数据读取和处理
2. 使用geom几何表进行精确对齐
3. 修复NaN相关性问题（从radargram提取表面功率）
4. 计算表面线位置误差RMSE
5. 三角交叉验证综合可视化
"""

import os
import numpy as np
import pvl
import pandas as pd
import matplotlib.pyplot as plt
from pds4_tools import pds4_read
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, pearsonr
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===========================
# 配置数据路径
# ===========================
# MOLA数据
MOLA_DIR = "MOLA MEGDR"
MOLA_N_IMG = os.path.join(MOLA_DIR, "megt00n090hb.img")  # -44° to 0°N
MOLA_N_LBL = os.path.join(MOLA_DIR, "megt00n090hb.lbl")
MOLA_S_IMG = os.path.join(MOLA_DIR, "megt44s090hb.img")  # -88° to -44°N
MOLA_S_LBL = os.path.join(MOLA_DIR, "megt44s090hb.lbl")

# SHARAD Clutter数据
CLUTTER_DIR = "SHARAD Clutter"
SIM_XML = os.path.join(CLUTTER_DIR, "s_00810101_sim.xml")
EMAP_XML = os.path.join(CLUTTER_DIR, "s_00810101_emap.xml")
RTRN_CSV = os.path.join(CLUTTER_DIR, "s_00810101_rtrn.csv")

# SHARAD Radargram数据（新增）
RADARGRAM_DIR = "RADARGRAM"
RGRAM_XML = os.path.join(RADARGRAM_DIR, "s_00810101_rgram.xml")
RGRAM_IMG = os.path.join(RADARGRAM_DIR, "s_00810101_rgram.img")

# SHARAD几何数据（新增）
GEOM_DIR = "GEOM"
GEOM_XML = os.path.join(GEOM_DIR, "s_00810101_geom.xml")
GEOM_TAB = os.path.join(GEOM_DIR, "s_00810101_geom.tab")

# 全局变量
DEM_N, META_N = None, None
DEM_S, META_S = None, None


# ===========================
# MOLA读取和处理函数（保持不变）
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
# 新增：Radargram读取函数
# ===========================
def read_radargram(rgram_xml, rgram_img=None):
    """读取SHARAD实测雷达图"""
    print(f"\n读取实测雷达图...")

    try:
        # 使用pds4_tools读取
        rg = pds4_read(rgram_xml)

        # 查找Array_2D_Image
        rg_arrs = [s for s in rg if hasattr(s, 'tag') and 'Array_2D_Image' in s.tag]
        if not rg_arrs:
            # 备选：直接取第一个数据数组
            rg_arrs = [s for s in rg if hasattr(s, 'data')]

        if not rg_arrs:
            raise ValueError("无法找到雷达图数据数组")

        R = rg_arrs[0]
        rg_img = R.data.astype(np.float32)

        # 应用缩放因子
        scale = getattr(R, 'scaling_factor', 1.0)
        offset = getattr(R, 'offset', 0.0)
        rg_img = rg_img * scale + offset

        # 尝试获取垂直采样间隔
        delta_t = None
        for k in ['line_sampling_interval', 'Line_Sampling_Interval',
                  'SAMPLE_TIME', 'sample_time']:
            try:
                if hasattr(R, 'meta_data') and k in R.meta_data:
                    delta_t = float(R.meta_data[k])
                    break
            except:
                pass

        # 默认值（SHARAD标准）
        if delta_t is None:
            delta_t = 0.375e-6  # 0.375 μs
            print(f"  警告：未找到采样间隔，使用默认值 {delta_t * 1e6:.3f} μs")

        print(f"  雷达图尺寸: {rg_img.shape}")
        print(f"  垂直采样间隔: {delta_t * 1e6:.3f} μs")
        print(f"  数据范围: [{np.min(rg_img):.2f}, {np.max(rg_img):.2f}]")

        return {
            'data': rg_img,
            'delta_t': delta_t,
            'metadata': R if hasattr(R, 'meta_data') else None
        }

    except Exception as e:
        print(f"  警告：无法读取雷达图 - {e}")

        # 备用方案：直接读取.img文件
        if rgram_img and os.path.exists(rgram_img):
            print(f"  尝试直接读取.img文件...")
            # 假设3600行 x 5551列，float32
            data = np.fromfile(rgram_img, dtype='>f4').reshape(3600, 5551)
            return {
                'data': data,
                'delta_t': 0.375e-6,
                'metadata': None
            }

        return None


# ===========================
# 新增：几何表读取函数
# ===========================
def read_geom_table(geom_xml, geom_tab=None):
    """读取SHARAD几何表"""
    print(f"\n读取几何表...")

    try:
        # 首先检查XML文件是否存在
        if not os.path.exists(geom_xml):
            print(f"  XML文件不存在: {geom_xml}")
            return None

        G = pds4_read(geom_xml, quiet=True, lazy_load=True)

        # 查找表格数据
        tables = [s for s in G if 'Table' in str(type(s))]
        if not tables:
            raise ValueError("未找到表格数据")

        tbl = tables[0]

        # 转换为pandas DataFrame
        if hasattr(tbl, 'to_table'):
            geom_table = tbl.to_table()
            geom_df = geom_table.to_pandas()
        else:
            # 备用：直接访问数据
            geom_df = pd.DataFrame(tbl.data)

        # 清理列名
        geom_df.columns = [c.strip() for c in geom_df.columns]

        print(f"  几何表尺寸: {len(geom_df)} 行")
        print(f"  可用列: {list(geom_df.columns)[:10]}...")

        return geom_df

    except Exception as e:
        print(f"  警告：无法从XML读取 - {e}")

        # 备用：尝试直接读取.tab文件
        if geom_tab and os.path.exists(geom_tab):
            print(f"  尝试直接读取.tab文件...")
            try:
                # 尝试多种分隔符
                for sep in ['\t', ',', ' ', None]:
                    try:
                        geom_df = pd.read_csv(geom_tab, sep=sep, engine='python')
                        if len(geom_df.columns) > 1:  # 确保成功分列
                            print(f"  几何表尺寸: {len(geom_df)} 行")
                            return geom_df
                    except:
                        continue
            except Exception as e2:
                print(f"  无法读取.tab文件: {e2}")

        return None


# ===========================
# 新增：表面功率提取函数
# ===========================
def extract_surface_power(radargram, geom_df, delta_t):
    """从实测雷达图提取表面功率"""
    print(f"\n提取表面功率...")

    rg_img = radargram['data']
    c0 = 299792458.0  # 光速

    # 1. 获取或计算表面线位置
    surf_bin = None

    # 尝试直接使用提供的表面行
    for cand in ['surface_line', 'SurfaceLine', 'NadirLine',
                 'surface_sample', 'Surface_Sample']:
        if cand in geom_df.columns:
            surf_bin = pd.to_numeric(geom_df[cand], errors='coerce').values
            print(f"  使用几何表字段: {cand}")
            break

    # 如果没有，用高度差计算
    if surf_bin is None:
        print(f"  根据高度差计算表面线...")

        # 获取高度数据
        h_sc = pd.to_numeric(geom_df.get('SpacecraftHgt', np.nan),
                             errors='coerce').values
        h_nad = pd.to_numeric(geom_df.get('NadirHgt', np.nan),
                              errors='coerce').values

        # 计算双程延迟
        rng = 2.0 * np.maximum(0, h_sc - h_nad)
        tau = rng / c0
        surf_bin = tau / delta_t

    # 2. 清理和限制范围
    cols = np.arange(min(len(surf_bin), rg_img.shape[1]))
    valid = ~np.isnan(surf_bin) & (surf_bin >= 0) & (surf_bin < rg_img.shape[0])
    surf_bin = np.clip(surf_bin, 0, rg_img.shape[0] - 1)

    # 3. 在表面线附近提取峰值
    W = 10  # 搜索窗口半宽
    peak_bin = np.full_like(surf_bin, np.nan, dtype=float)
    peak_pow = np.full_like(surf_bin, np.nan, dtype=float)

    for j in cols[valid[:len(cols)]]:
        i0 = int(round(surf_bin[j]))
        i1 = max(0, i0 - W)
        i2 = min(rg_img.shape[0], i0 + W + 1)

        col = rg_img[i1:i2, j]
        if col.size == 0:
            continue

        # 找到最大值位置
        k = np.argmax(np.abs(col))  # 使用绝对值
        peak_bin[j] = i1 + k
        peak_pow[j] = np.abs(col[k])

    # 4. 计算对齐误差
    valid_mask = ~np.isnan(peak_bin) & ~np.isnan(surf_bin[:len(peak_bin)])
    if np.sum(valid_mask) > 0:
        bin_err = peak_bin[valid_mask] - surf_bin[:len(peak_bin)][valid_mask]
        rmse_bins = np.sqrt(np.mean(bin_err ** 2))
        rmse_m = rmse_bins * (c0 * delta_t) / 2.0

        print(f"  表面线对齐RMSE: {rmse_bins:.2f} 行 (~{rmse_m:.2f} m)")
        print(f"  平均偏移: {np.mean(bin_err):.2f} 行")
        print(f"  标准差: {np.std(bin_err):.2f} 行")
    else:
        bin_err = np.array([])
        rmse_bins = rmse_m = np.nan

    return {
        'surf_bin': surf_bin,
        'peak_bin': peak_bin,
        'peak_pow': peak_pow,
        'bin_err': bin_err,
        'rmse_bins': rmse_bins,
        'rmse_m': rmse_m
    }


# ===========================
# 修正：相关性分析函数
# ===========================
def compute_correlations(mola_elev, peak_pow, geom_df):
    """计算MOLA高程与雷达功率的相关性"""
    print(f"\n计算相关性分析...")

    # 确保长度匹配
    min_len = min(len(mola_elev), len(peak_pow))
    mola_subset = mola_elev[:min_len]
    power_subset = peak_pow[:min_len]

    # 有效数据掩码
    valid = (~np.isnan(mola_subset)) & (~np.isnan(power_subset)) & (power_subset > 0)

    if np.sum(valid) < 100:
        print(f"  警告：有效数据点不足 ({np.sum(valid)})")
        return None

    # 提取有效数据
    mola_valid = mola_subset[valid]
    power_valid = power_subset[valid]

    # 检查数据变异性
    if np.std(power_valid) < 1e-12:
        print(f"  警告：功率数据变异性太小，无法计算相关性")
        return None

    # 对功率取对数（避免极值影响）
    power_db = 10 * np.log10(power_valid + 1e-10)

    results = {}

    try:
        # Spearman相关（更稳健）
        rho_s, p_s = spearmanr(mola_valid, power_db)
        results['spearman_r'] = rho_s
        results['spearman_p'] = p_s
        print(f"  Spearman ρ = {rho_s:.3f} (p = {p_s:.2e})")

        # Pearson相关
        r_p, p_p = pearsonr(mola_valid, power_db)
        results['pearson_r'] = r_p
        results['pearson_p'] = p_p
        print(f"  Pearson r = {r_p:.3f} (p = {p_p:.2e})")

        # 计算地形粗糙度与功率的相关
        window = 50
        roughness = np.zeros(len(mola_valid))
        for i in range(len(mola_valid)):
            start = max(0, i - window // 2)
            end = min(len(mola_valid), i + window // 2 + 1)
            if end > start:
                roughness[i] = np.std(mola_valid[start:end])

        rho_rough, p_rough = spearmanr(roughness, power_db)
        results['roughness_r'] = rho_rough
        results['roughness_p'] = p_rough
        print(f"  粗糙度-功率相关: ρ = {rho_rough:.3f} (p = {p_rough:.2e})")

    except Exception as e:
        print(f"  相关性计算失败: {e}")
        return None

    results['n_valid'] = np.sum(valid)
    results['mola_valid'] = mola_valid
    results['power_db'] = power_db

    return results


# ===========================
# 综合可视化函数
# ===========================
def create_triangular_validation_plot(radargram, clutter, mola_elev,
                                      surface_extraction, correlation_results,
                                      geom_df, output_name="triangular_validation.png"):
    """创建三角交叉验证综合图"""

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(5, 3, figure=fig,
                  height_ratios=[1.2, 1.2, 0.8, 0.6, 0.8],
                  hspace=0.3, wspace=0.25)

    # 准备数据
    rg_img = radargram['data']
    sim = clutter['sim']
    surf_bin = surface_extraction['surf_bin']
    peak_bin = surface_extraction['peak_bin']
    peak_pow = surface_extraction['peak_pow']
    bin_err = surface_extraction['bin_err']

    # 1. 实测雷达图
    ax1 = fig.add_subplot(gs[0, :])

    # 对数尺度显示
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

    if correlation_results is not None:
        mola_valid = correlation_results['mola_valid']
        power_db = correlation_results['power_db']

        # 密度散点图
        from scipy.stats import gaussian_kde

        # 下采样以提高性能
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
        z = np.polyfit(mola_valid, power_db, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(np.min(mola_valid), np.max(mola_valid), 100)
        ax6.plot(x_trend, p(x_trend), 'r-', alpha=0.5, linewidth=2)

        ax6.set_xlabel("MOLA Elevation (m)")
        ax6.set_ylabel("Echo Power (dB)")
        ax6.set_title(f"Elevation vs Power\nρ={correlation_results['spearman_r']:.3f}, "
                      f"r={correlation_results['pearson_r']:.3f}",
                      fontsize=11)
        ax6.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax6, label="Density")

    # 7. 轨道地图
    ax7 = fig.add_subplot(gs[4, 2])

    if 'FirstLon' in geom_df.columns and 'FirstLat' in geom_df.columns:
        lons = geom_df['FirstLon'].values
        lats = geom_df['FirstLat'].values

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
    plt.suptitle("SHARAD-MOLA-Radargram Triangular Cross-Validation",
                 fontsize=14, fontweight='bold', y=0.995)

    # 添加汇总文本框
    if correlation_results:
        spearman_str = f"{correlation_results['spearman_r']:.3f}"
        pearson_str = f"{correlation_results['pearson_r']:.3f}"
        valid_points = correlation_results['n_valid']
    else:
        spearman_str = "N/A"
        pearson_str = "N/A"
        valid_points = 0

    summary_text = f"""
    === Validation Metrics ===
    Surface RMSE: {surface_extraction['rmse_bins']:.2f} bins ({surface_extraction['rmse_m']:.1f} m)
    Valid Points: {valid_points}
    Spearman ρ: {spearman_str}
    Pearson r: {pearson_str}
    Coverage: {np.sum(~np.isnan(mola_elev)) / len(mola_elev) * 100:.1f}%
    """

    fig.text(0.99, 0.01, summary_text.strip(),
             horizontalalignment='right', verticalalignment='bottom',
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\n✓ 综合验证图已保存: {output_name}")
    plt.show()


# ===========================
# 主程序：三角交叉验证
# ===========================
def triangular_cross_validation():
    """执行完整的三角交叉验证"""

    print("=" * 80)
    print("SHARAD-MOLA-Radargram 三角交叉验证")
    print("=" * 80)

    # 1. 读取MOLA数据
    print("\n[步骤1] 读取MOLA MEGDR瓦片")
    global DEM_N, META_N, DEM_S, META_S

    if os.path.exists(MOLA_N_IMG):
        DEM_N, META_N = read_megdr(MOLA_N_IMG, MOLA_N_LBL)
    else:
        print(f"  警告：北瓦片不存在")

    if os.path.exists(MOLA_S_IMG):
        DEM_S, META_S = read_megdr(MOLA_S_IMG, MOLA_S_LBL)
    else:
        print(f"  警告：南瓦片不存在")

    # 2. 读取官方杂波数据
    print("\n[步骤2] 读取官方杂波模拟")
    clutter_data = read_sharad_clutter(SIM_XML, EMAP_XML, RTRN_CSV)

    if 'rtrn' not in clutter_data:
        print("错误：无RTRN数据")
        return

    # 3. 读取实测雷达图
    print("\n[步骤3] 读取实测雷达图")

    if not os.path.exists(RGRAM_XML):
        print(f"错误：雷达图文件不存在: {RGRAM_XML}")
        print("请下载 s_00810101_rgram.xml 和 s_00810101_rgram.img")
        return

    radargram = read_radargram(RGRAM_XML, RGRAM_IMG)
    if radargram is None:
        print("错误：无法读取雷达图")
        return

    # 4. 读取几何表
    print("\n[步骤4] 读取几何表")

    if not os.path.exists(GEOM_XML):
        print(f"警告：几何表不存在，将使用RTRN数据")
        geom_df = clutter_data['rtrn']
    else:
        geom_df = read_geom_table(GEOM_XML, GEOM_TAB)
        if geom_df is None:
            print("警告：无法读取几何表，使用RTRN数据")
            geom_df = clutter_data['rtrn']

    # 5. 沿轨采样MOLA高程
    print("\n[步骤5] 沿轨采样MOLA高程")

    # 选择经纬度源
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

    if not lon_col or not lat_col:
        print("错误：无法找到经纬度列")
        return

    print(f"  使用经度源: {lon_col}")
    print(f"  使用纬度源: {lat_col}")

    # 采样
    mola_elev = []
    for idx, (lonE, lat) in enumerate(zip(geom_df[lon_col].values,
                                          geom_df[lat_col].values)):
        if lonE < 0:
            lonE += 360
        elev = sample_mola(float(lonE), float(lat))
        mola_elev.append(elev)

        if (idx + 1) % 1000 == 0:
            valid_count = np.sum(~np.isnan(mola_elev))
            pct = valid_count / (idx + 1) * 100
            print(f"  进度: {idx + 1}/{len(geom_df)}, 有效率: {pct:.1f}%")

    mola_elev = np.array(mola_elev)

    # 6. 提取表面功率
    print("\n[步骤6] 从雷达图提取表面功率")
    surface_extraction = extract_surface_power(radargram, geom_df,
                                               radargram['delta_t'])

    # 7. 计算相关性
    print("\n[步骤7] 计算相关性分析")
    correlation_results = compute_correlations(mola_elev,
                                               surface_extraction['peak_pow'],
                                               geom_df)

    # 8. 创建综合可视化
    print("\n[步骤8] 生成三角交叉验证图")
    create_triangular_validation_plot(radargram, clutter_data, mola_elev,
                                      surface_extraction, correlation_results,
                                      geom_df)

    # 9. 生成验证报告
    print("\n" + "=" * 80)
    print("验证报告汇总")
    print("=" * 80)

    print(f"\n数据覆盖率:")
    print(f"  MOLA采样: {np.sum(~np.isnan(mola_elev)) / len(mola_elev) * 100:.1f}%")
    print(
        f"  表面检测: {np.sum(~np.isnan(surface_extraction['peak_pow'])) / len(surface_extraction['peak_pow']) * 100:.1f}%")

    print(f"\n对齐精度:")
    print(f"  表面线RMSE: {surface_extraction['rmse_bins']:.2f} bins")
    print(f"  等效距离: {surface_extraction['rmse_m']:.1f} m")

    if correlation_results:
        print(f"\n相关性分析:")
        print(f"  高程-功率 Spearman: ρ = {correlation_results['spearman_r']:.3f}")
        print(f"  高程-功率 Pearson: r = {correlation_results['pearson_r']:.3f}")
        print(f"  粗糙度-功率: ρ = {correlation_results['roughness_r']:.3f}")
        print(f"  有效数据点: {correlation_results['n_valid']}")

    print("\n✓ 三角交叉验证完成！")

    return {
        'radargram': radargram,
        'clutter': clutter_data,
        'mola_elev': mola_elev,
        'surface': surface_extraction,
        'correlation': correlation_results,
        'geom': geom_df
    }


# ===========================
# 辅助函数：读取杂波数据
# ===========================
def read_sharad_clutter(sim_xml, emap_xml=None, rtrn_csv=None):
    """读取SHARAD Clutter套件数据（保持原有功能）"""
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
# 批量处理多条轨道
# ===========================
def batch_process_tracks(track_list):
    """批量处理多条轨道进行交叉验证"""

    results_summary = []

    for track_id in track_list:
        print(f"\n处理轨道: {track_id}")
        print("-" * 60)

        # 更新文件路径
        global SIM_XML, EMAP_XML, RTRN_CSV, RGRAM_XML, RGRAM_IMG, GEOM_XML, GEOM_TAB

        SIM_XML = os.path.join(CLUTTER_DIR, f"{track_id}_sim.xml")
        EMAP_XML = os.path.join(CLUTTER_DIR, f"{track_id}_emap.xml")
        RTRN_CSV = os.path.join(CLUTTER_DIR, f"{track_id}_rtrn.csv")
        RGRAM_XML = os.path.join(RADARGRAM_DIR, f"{track_id}_rgram.xml")
        RGRAM_IMG = os.path.join(RADARGRAM_DIR, f"{track_id}_rgram.img")
        GEOM_XML = os.path.join(GEOM_DIR, f"{track_id}_geom.xml")
        GEOM_TAB = os.path.join(GEOM_DIR, f"{track_id}_geom.tab")

        # 检查文件是否存在
        required_files = [SIM_XML, RTRN_CSV]
        if not all(os.path.exists(f) for f in required_files):
            print(f"  跳过：缺少必需文件")
            continue

        try:
            # 执行验证
            results = triangular_cross_validation()

            if results and results['surface']:
                # 收集关键指标
                summary = {
                    'track_id': track_id,
                    'coverage': np.sum(~np.isnan(results['mola_elev'])) / len(results['mola_elev']) * 100,
                    'rmse_bins': results['surface']['rmse_bins'],
                    'rmse_m': results['surface']['rmse_m'],
                    'spearman_r': results['correlation']['spearman_r'] if results['correlation'] else np.nan,
                    'pearson_r': results['correlation']['pearson_r'] if results['correlation'] else np.nan,
                    'n_valid': results['correlation']['n_valid'] if results['correlation'] else 0
                }
                results_summary.append(summary)

        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 生成汇总表
    if results_summary:
        df_summary = pd.DataFrame(results_summary)
        print("\n" + "=" * 80)
        print("批量处理汇总")
        print("=" * 80)
        print(df_summary.to_string())

        # 保存到CSV
        df_summary.to_csv("validation_summary.csv", index=False)
        print("\n✓ 汇总结果已保存到 validation_summary.csv")

    return results_summary


# ===========================
# 主入口
# ===========================
if __name__ == "__main__":
    try:
        # 单轨验证
        print("执行单轨三角交叉验证...")
        results = triangular_cross_validation()

        # 可选：批量处理多条轨道
        # track_list = ['s_00810101', 's_00810102', 's_00810103']
        # batch_results = batch_process_tracks(track_list)

        print("\n所有分析完成！")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()