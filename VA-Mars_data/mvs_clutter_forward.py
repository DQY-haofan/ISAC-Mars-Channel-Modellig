#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
火星ISAC信道模型验证框架 V3.1 - 修正版
重点修正：
1. K因子按表面ROI重算（±20-30 bins窗口）
2. 大气衰减功率/幅度口径统一
3. 坡度计算加入cos(φ)纬度修正
4. 信道分解图独立色轴

对应论文Section II: The Unified Mars ISAC Channel Model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.ndimage import sobel, uniform_filter, gaussian_filter
from scipy.stats import gaussian_kde, spearmanr
from scipy.interpolate import interp1d
import time
import warnings

warnings.filterwarnings('ignore')


# ===========================
# 图像质量评估指标
# ===========================

def structural_similarity(im1, im2, data_range=1.0, win_size=7):
    """计算SSIM（结构相似性指数）"""
    im1 = np.asarray(im1, dtype=np.float64)
    im2 = np.asarray(im2, dtype=np.float64)

    if im1.shape != im2.shape:
        raise ValueError("输入图像必须有相同的尺寸")

    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = uniform_filter(im1, size=win_size)
    mu2 = uniform_filter(im2, size=win_size)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(im1 * im1, size=win_size) - mu1_sq
    sigma2_sq = uniform_filter(im2 * im2, size=win_size) - mu2_sq
    sigma12 = uniform_filter(im1 * im2, size=win_size) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return np.mean(ssim_map)


def peak_signal_noise_ratio(image_true, image_test, data_range=1.0):
    """计算PSNR（峰值信噪比）"""
    image_true = np.asarray(image_true, dtype=np.float64)
    image_test = np.asarray(image_test, dtype=np.float64)

    if image_true.shape != image_test.shape:
        raise ValueError("输入图像必须有相同的尺寸")

    mse = np.mean((image_true - image_test) ** 2)

    if mse == 0:
        return float('inf')

    max_pixel = data_range
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


# ===========================
# 物理常量
# ===========================
C0 = 299792458.0  # 光速 m/s
FC = 20e6  # SHARAD中心频率 20 MHz
LAMBDA = C0 / FC  # 波长 λ ~15m
MARS_R = 3389500.0  # 火星平均半径 m
DEG = np.pi / 180.0  # 度转弧度
K = 2 * np.pi / LAMBDA  # 波数

# MOLA参数
MEG128_PPD = 128  # 每度像素数
MEG128_PIXEL_M = 463.0  # 像素间距

# 天线参数
ANTENNA_3DB = 3.0 * DEG  # 3dB波束宽度
PSI0 = ANTENNA_3DB / np.sqrt(np.log(2))  # 高斯波束参数

# ===========================
# 仿真质量控制
# ===========================
FAST_MODE = True  # 快速模式开关
FULL_RES_EVAL = True  # 全分辨率评估开关

if FAST_MODE:
    SIM = {
        "col_subsample": 20,  # 列下采样（仿真时）
        "patch_size": 21,  # patch大小
        "patch_km": 10.0,  # patch范围km
        "gate_bins": 300,  # NadirLine附近的有效窗口
        "param_steps": 7,  # 参数扫描步数
        "eval_window": 120,  # 评估窗口半宽（bins）
        "spec_sigma": 15.0 * DEG,  # 镜面高斯宽度
        "eval_subsample": 1 if FULL_RES_EVAL else 20,  # 评估时的下采样率
        "k_factor_window": 25  # K因子计算窗口（修正点1）
    }
else:
    SIM = {
        "col_subsample": 5,
        "patch_size": 51,
        "patch_km": 25.0,
        "gate_bins": 600,
        "param_steps": 15,
        "eval_window": 200,
        "spec_sigma": 10.0 * DEG,
        "eval_subsample": 1,
        "k_factor_window": 30  # K因子计算窗口（修正点1）
    }


# ===========================
# 环境参数向量
# ===========================
class EnvironmentalParameters:
    """统一环境状态向量 θ_env"""

    def __init__(self):
        self.tau_vis = 0.5  # 尘埃光学深度
        self.epsilon_r = 4.0  # 表面介电常数（实部）
        self.sigma_h = 0.05  # 表面RMS高度 (m)
        self.correlation_length = 1.0  # 相关长度 (m)
        self.S4 = 0.3  # 电离层闪烁指数
        self.Cn2_surface = 1e-14  # 地表折射率结构常数
        self.k0 = 0.005  # 漫散射基底系数（基准值）
        self.n_lambert = 1.5  # Lambert指数

    def to_vector(self):
        """转换为向量形式"""
        return np.array([self.tau_vis, self.epsilon_r, self.sigma_h,
                         self.correlation_length, self.S4])

    def from_vector(self, vec):
        """从向量恢复"""
        self.tau_vis = vec[0]
        self.epsilon_r = vec[1]
        self.sigma_h = vec[2]
        self.correlation_length = vec[3]
        self.S4 = vec[4]


# ===========================
# 地表类型定义
# ===========================
SURFACE_TYPES = {
    'basalt': {
        'epsilon_r': 6.0,
        'sigma_h': 0.02,
        'correlation_length': 0.5,
        'w_spec': 0.8,  # 镜面权重
        'k0': 0.003  # 漫反射基底系数
    },
    'aeolian': {
        'epsilon_r': 2.5,
        'sigma_h': 0.10,
        'correlation_length': 2.0,
        'w_spec': 0.2,
        'k0': 0.008
    }
}

# 文件路径
DATA_DIR = {
    'mola': "MOLA MEGDR",
    'clutter': "SHARAD Clutter",
    'radargram': "RADARGRAM",
    'geom': "GEOM"
}


# ===========================
# 理论模型实现
# ===========================

def fresnel_reflection_coefficient(theta_i, epsilon_r, polarization='H'):
    """计算Fresnel反射系数"""
    cos_theta = np.cos(theta_i)
    sin_theta = np.sin(theta_i)

    n = np.sqrt(epsilon_r + 0j)
    cos_t = np.sqrt(1 - (sin_theta / n) ** 2 + 0j)

    if polarization == 'H':
        r = (cos_theta - n * cos_t) / (cos_theta + n * cos_t)
    else:
        r = (n * cos_theta - cos_t) / (n * cos_theta + cos_t)

    return np.abs(r) ** 2


def effective_reflection_coefficient(theta_i, epsilon_r, sigma_h, lambda_=LAMBDA):
    """有效反射系数，包含粗糙度效应"""
    Gamma = fresnel_reflection_coefficient(theta_i, epsilon_r)
    roughness_factor = np.exp(-(4 * np.pi * sigma_h * np.cos(theta_i) / lambda_) ** 2)
    return Gamma * roughness_factor


def select_scattering_model(k, sigma_h, correlation_length):
    """混合散射模型选择"""
    kl = k * correlation_length
    k_sigma = k * sigma_h

    if kl > 6 and correlation_length ** 2 > 2.76 * sigma_h * LAMBDA:
        return 'KA'
    elif k_sigma < 0.3:
        return 'SPM'
    else:
        return 'IEM'


def compute_diffuse_scattering_enhanced(theta_i, sigma_h, correlation_length, epsilon_r,
                                        k0=0.005, n_lambert=1.5):
    """
    改进的漫散射功率计算（补丁5：k_d带入角度和粗糙度调制）
    """
    model = select_scattering_model(K, sigma_h, correlation_length)

    # 角度和粗糙度调制的k_d
    k_d = k0 * np.cos(theta_i) ** n_lambert * \
          np.exp(-(4 * np.pi * sigma_h * np.cos(theta_i) / LAMBDA) ** 2)

    if model == 'SPM':
        cos_theta = np.cos(theta_i)
        sin_theta = np.sin(theta_i)

        sigma0_diff = (K ** 4 * sigma_h ** 2 * correlation_length ** 2 / np.pi) * \
                      np.exp(-(K * correlation_length * sin_theta) ** 2) * \
                      np.abs((epsilon_r - 1) / (epsilon_r + 1)) ** 2 * cos_theta ** 4

    elif model == 'KA':
        sigma0_diff = 0.1 * np.cos(theta_i)

    else:  # IEM
        sigma0_diff = 0.05 * np.cos(theta_i)

    # 添加调制后的朗伯底座
    sigma0_diff = sigma0_diff + k_d

    return sigma0_diff


def compute_rician_k_factor_surface_roi(H_spec, H_diff, nadir_line, window=25):
    """
    修正点1：基于表面ROI计算Rician K因子
    只在预测表面±window bins内计算
    K = μ_spec^2 / (2σ_diff^2)
    """
    nadir = int(nadir_line)

    # 定义表面ROI窗口
    roi_start = max(0, nadir - window)
    roi_end = min(len(H_spec), nadir + window)

    if roi_end - roi_start < 5:  # 窗口太小
        return 0, -np.inf

    # 只在ROI内取镜面和漫散射分量
    H_spec_roi = H_spec[roi_start:roi_end]
    H_diff_roi = H_diff[roi_start:roi_end]

    # 镜面分量的幅度平均
    mu_spec = np.mean(np.abs(H_spec_roi))

    # 漫散射分量的方差
    diff_abs = np.abs(H_diff_roi)
    sigma_diff2 = np.var(diff_abs)

    if sigma_diff2 > 0:
        K = (mu_spec ** 2) / (2 * sigma_diff2)
        K_dB = 10 * np.log10(K + 1e-10)
    else:
        K = 0
        K_dB = -np.inf

    return K, K_dB


def atmospheric_attenuation(distance, tau_vis, frequency=FC, return_power=True):
    """
    修正点2：大气衰减模型，明确返回功率或幅度因子

    Args:
        return_power: True返回功率因子(10^(-L/10))，False返回幅度因子(10^(-L/20))
    """
    A_dust = 0.1 * tau_vis * (frequency / 1e9) ** 0.5
    L_atm = A_dust * (distance / 1000.0)  # dB

    if return_power:
        return 10 ** (-L_atm / 10.0)  # 功率因子
    else:
        return 10 ** (-L_atm / 20.0)  # 幅度因子


def compute_specular_weight_simplified(theta_i, slope, spec_sigma=5.0 * DEG):
    """简化的镜面权重计算"""
    slope_factor = np.exp(-(slope / 0.05) ** 2)
    angle_factor = np.exp(-(theta_i ** 2) / (spec_sigma ** 2))
    w_spec = slope_factor * angle_factor
    return w_spec


# ===========================
# MOLA数据处理
# ===========================

def read_mola_tile(img_path, lbl_path):
    """读取MOLA瓦片"""
    import pvl

    lbl = pvl.load(lbl_path)
    lines = int(lbl['IMAGE']['LINES'])
    cols = int(lbl['IMAGE']['LINE_SAMPLES'])

    data = np.fromfile(img_path, dtype='>i2').reshape(lines, cols)
    data = data.astype(np.float32)
    data[data == -32768] = np.nan

    proj = lbl['IMAGE_MAP_PROJECTION']
    meta = {
        'data': data,
        'lon_min': float(proj['WESTERNMOST_LONGITUDE']),
        'lon_max': float(proj['EASTERNMOST_LONGITUDE']),
        'lat_min': float(proj['MINIMUM_LATITUDE']),
        'lat_max': float(proj['MAXIMUM_LATITUDE']),
        'ppd': MEG128_PPD
    }

    return meta


MOLA_TILES = []


def load_mola():
    """加载所有MOLA瓦片"""
    global MOLA_TILES

    tiles = [
        ('megt00n090hb.img', 'megt00n090hb.lbl'),
        ('megt44s090hb.img', 'megt44s090hb.lbl')
    ]

    print(f"MOLA数据目录: {DATA_DIR['mola']}")

    for img_file, lbl_file in tiles:
        img_path = os.path.join(DATA_DIR['mola'], img_file)
        lbl_path = os.path.join(DATA_DIR['mola'], lbl_file)

        if os.path.exists(img_path) and os.path.exists(lbl_path):
            try:
                tile = read_mola_tile(img_path, lbl_path)
                MOLA_TILES.append(tile)
                print(f"  ✓ 成功加载: {img_file}")
            except Exception as e:
                print(f"  ✗ 加载失败: {e}")

    if len(MOLA_TILES) == 0:
        print("\n警告：无MOLA数据")
    else:
        print(f"\n成功加载 {len(MOLA_TILES)} 个MOLA瓦片")


def sample_mola(lat, lon):
    """从MOLA采样高程"""
    lon = lon % 360.0

    for tile in MOLA_TILES:
        if (tile['lon_min'] <= lon <= tile['lon_max'] and
                tile['lat_min'] <= lat <= tile['lat_max']):

            j = (lon - tile['lon_min']) * tile['ppd']
            i = (tile['lat_max'] - lat) * tile['ppd']

            if not (0 <= j < tile['data'].shape[1] - 1) or \
                    not (0 <= i < tile['data'].shape[0] - 1):
                return np.nan

            i0, j0 = int(i), int(j)
            di, dj = i - i0, j - j0

            v00 = tile['data'][i0, j0]
            v01 = tile['data'][i0, j0 + 1]
            v10 = tile['data'][i0 + 1, j0]
            v11 = tile['data'][i0 + 1, j0 + 1]

            if np.isnan([v00, v01, v10, v11]).any():
                return np.nan

            return (1 - di) * (1 - dj) * v00 + (1 - di) * dj * v01 + \
                   di * (1 - dj) * v10 + di * dj * v11

    return np.nan


# ===========================
# 增强版MVS前向仿真器V3.1
# ===========================

def forward_simulation_v3(geom_df, n_rows, n_cols, dt, env_params, full_resolution=False):
    """
    前向仿真V3.1，修正了K因子计算、大气衰减和坡度计算

    主要修正：
    1. K因子基于表面ROI计算
    2. 大气衰减使用功率因子
    3. 坡度计算加入cos(φ)修正
    """
    print("\n执行理论模型前向仿真V3.1...")

    # 决定是否下采样
    if full_resolution:
        subsample = 1
        print(f"  模式: 全分辨率")
    else:
        subsample = SIM['col_subsample']
        print(f"  模式: {'快速' if FAST_MODE else '高精度'}")
        print(f"  列下采样: 1/{subsample}")

    print(f"  Patch: {SIM['patch_size']}×{SIM['patch_size']} ({SIM['patch_km']}×{SIM['patch_km']} km)")

    # 初始化信道分量
    H_spec = np.zeros((n_rows, n_cols), dtype=complex)
    H_diff = np.zeros((n_rows, n_cols), dtype=complex)

    # 存储每列的NadirLine
    nadir_lines = np.zeros(n_cols)

    # 列下采样
    cols_to_process = np.arange(0, min(n_cols, len(geom_df)), subsample)

    # 统计
    valid_facets = 0
    spec_facets = 0

    # 用于插值的临时存储
    H_spec_sparse = {}
    H_diff_sparse = {}

    for idx, j in enumerate(cols_to_process):
        if idx % 50 == 0:
            print(f"  进度: {idx}/{len(cols_to_process)}", end='\r')

        # 获取几何信息
        try:
            sc_lon = float(geom_df.iloc[j].get('SpacecraftLon', 140))
            sc_lat = float(geom_df.iloc[j].get('SpacecraftLat', -45))
            sc_hgt = float(geom_df.iloc[j].get('SpacecraftHgt', 3000))

            gnd_lon = float(geom_df.iloc[j].get('FirstLon', sc_lon))
            gnd_lat = float(geom_df.iloc[j].get('FirstLat', sc_lat))

            nadir_line = float(geom_df.iloc[j].get('NadirLine', 1800))
            nadir_hgt = float(geom_df.iloc[j].get('NadirHgt', 0))

            if np.isnan(sc_hgt) or sc_hgt < 100 or sc_hgt > 10000:
                sc_hgt = 3000
            if np.isnan(nadir_line):
                nadir_line = 1800

            nadir_lines[j] = nadir_line

        except Exception as e:
            continue

        if sc_lon < 0: sc_lon += 360
        if gnd_lon < 0: gnd_lon += 360

        # 计算双程距离
        R_nadir = sc_hgt - nadir_hgt
        tau_nadir = 2 * R_nadir / C0

        # 采样局部DEM
        patch_size = SIM['patch_size']
        dlat = SIM['patch_km'] / 111.0

        # 修正点3：加入cos(φ)纬度修正
        cos_phi = np.cos(gnd_lat * DEG)
        dlon = dlat / max(0.001, cos_phi)

        lats = np.linspace(gnd_lat - dlat / 2, gnd_lat + dlat / 2, patch_size)
        lons = np.linspace(gnd_lon - dlon / 2, gnd_lon + dlon / 2, patch_size)

        # 临时存储该列的信道响应
        H_spec_col = np.zeros(n_rows, dtype=complex)
        H_diff_col = np.zeros(n_rows, dtype=complex)

        # 处理patch
        step = 2 if FAST_MODE and not full_resolution else 1
        for ii in range(0, patch_size, step):
            for jj in range(0, patch_size, step):
                facet_lat = lats[ii]
                facet_lon = lons[jj]
                facet_elev = sample_mola(facet_lat, facet_lon)

                if np.isnan(facet_elev):
                    facet_elev = 0

                # 几何计算
                dlat_rad = (facet_lat - sc_lat) * DEG
                dlon_rad = (facet_lon - sc_lon) * DEG * np.cos(sc_lat * DEG)

                d_horiz = MARS_R * np.sqrt(dlat_rad ** 2 + dlon_rad ** 2)
                d_vert = (sc_hgt - facet_elev)
                R = np.sqrt(d_horiz ** 2 + d_vert ** 2)

                if R < 100 or R > 10000:
                    continue

                theta_i = np.arctan(d_horiz / max(100, d_vert))
                theta_i = np.clip(theta_i, 0, 80 * DEG)

                # 修正点3：计算局部坡度时考虑纬度效应
                slope = 0.01
                if not np.isnan(facet_elev):
                    # 经度方向梯度（考虑纬度修正）
                    elev_dx = sample_mola(facet_lat, facet_lon + 0.001)
                    # 纬度方向梯度
                    elev_dy = sample_mola(facet_lat + 0.001, facet_lon)

                    if not np.isnan(elev_dx) and not np.isnan(elev_dy):
                        # 修正：经度方向的距离要乘以cos(φ)
                        dx_m = (elev_dx - facet_elev)
                        dy_m = (elev_dy - facet_elev)
                        # 正确的梯度计算
                        dx_km = 0.001 * 111.0 * cos_phi  # 经度差转km（带纬度修正）
                        dy_km = 0.001 * 111.0  # 纬度差转km
                        slope = np.sqrt((dx_m / dx_km) ** 2 + (dy_m / dy_km) ** 2) / 1000.0

                        if slope > 0.03:
                            surface = SURFACE_TYPES['basalt']
                        else:
                            surface = SURFACE_TYPES['aeolian']
                    else:
                        surface = SURFACE_TYPES['aeolian']
                else:
                    surface = SURFACE_TYPES['aeolian']

                # 镜面权重
                w_spec = compute_specular_weight_simplified(
                    theta_i, slope, spec_sigma=SIM['spec_sigma']
                )

                # 镜面反射
                Gamma_eff = effective_reflection_coefficient(
                    theta_i, surface['epsilon_r'], surface['sigma_h'])
                P_spec_raw = Gamma_eff * w_spec

                if w_spec > 0.1:
                    spec_facets += 1

                # 改进的漫散射（使用增强模型）
                sigma0_diff = compute_diffuse_scattering_enhanced(
                    theta_i, surface['sigma_h'],
                    surface['correlation_length'],
                    surface['epsilon_r'],
                    k0=surface['k0'],
                    n_lambert=env_params.n_lambert
                )

                # 修正点2：大气衰减返回功率因子
                L_atm_power = atmospheric_attenuation(R, env_params.tau_vis, return_power=True)

                # 面元面积
                facet_area = (SIM['patch_km'] * 1000 / patch_size) ** 2

                # 功率计算（使用功率因子）
                P_spec = (LAMBDA ** 2 / (4 * np.pi) ** 3) * (1.0 / R ** 4) * \
                         P_spec_raw * facet_area * L_atm_power

                P_diff = (LAMBDA ** 2 / (4 * np.pi) ** 3) * (1.0 / R ** 4) * \
                         sigma0_diff * facet_area * L_atm_power

                # 时延计算
                tau = 2 * R / C0
                dtau = tau - tau_nadir
                dk = int(round(dtau / dt))

                if abs(dk) > SIM['gate_bins']:
                    continue

                k = int(nadir_line) + dk

                if 0 <= k < n_rows:
                    beam_weight = np.exp(-(dk / (SIM['gate_bins'] / 3)) ** 2)

                    phase_spec = np.exp(-1j * 2 * np.pi * FC * tau)
                    phase_diff = np.exp(-1j * 2 * np.pi * FC * tau) * \
                                 np.exp(1j * 2 * np.pi * np.random.rand())

                    H_spec_col[k] += np.sqrt(P_spec) * phase_spec * beam_weight
                    H_diff_col[k] += np.sqrt(P_diff) * phase_diff * beam_weight

                    valid_facets += 1

        # 存储处理的列
        H_spec[:, j] = H_spec_col
        H_diff[:, j] = H_diff_col

        H_spec_sparse[j] = H_spec_col
        H_diff_sparse[j] = H_diff_col

    print(f"\n  有效面元: {valid_facets}")
    print(f"  镜面面元: {spec_facets} ({100 * spec_facets / (valid_facets + 1):.1f}%)")

    # 插值填充（仅在非全分辨率模式下）
    if not full_resolution and subsample > 1:
        print("  插值填充未处理列...")
        processed_cols = sorted(H_spec_sparse.keys())

        if len(processed_cols) > 1:
            f_nadir = interp1d(processed_cols,
                               [nadir_lines[col] for col in processed_cols],
                               kind='linear', fill_value='extrapolate')

            for j in range(n_cols):
                if j not in H_spec_sparse:
                    left_idx = np.searchsorted(processed_cols, j) - 1
                    right_idx = min(left_idx + 1, len(processed_cols) - 1)
                    left_idx = max(0, left_idx)

                    if left_idx == right_idx:
                        nearest = processed_cols[left_idx]
                        H_spec[:, j] = H_spec[:, nearest]
                        H_diff[:, j] = H_diff[:, nearest]
                    else:
                        left_col = processed_cols[left_idx]
                        right_col = processed_cols[right_idx]
                        alpha = (j - left_col) / (right_col - left_col)

                        H_spec[:, j] = (1 - alpha) * H_spec_sparse[left_col] + \
                                       alpha * H_spec_sparse[right_col]
                        H_diff[:, j] = (1 - alpha) * H_diff_sparse[left_col] + \
                                       alpha * H_diff_sparse[right_col]

                    nadir_lines[j] = f_nadir(j)

    # 避免全零
    if np.max(np.abs(H_spec)) == 0:
        H_spec += (np.random.randn(n_rows, n_cols) + 1j * np.random.randn(n_rows, n_cols)) * 1e-6
    if np.max(np.abs(H_diff)) == 0:
        H_diff += (np.random.randn(n_rows, n_cols) + 1j * np.random.randn(n_rows, n_cols)) * 1e-6

    # 合成总信道
    H_total = H_spec + H_diff
    clutter = np.abs(H_total) ** 2

    # 修正点1：使用表面ROI计算K因子
    K_map = np.zeros(n_cols)
    for j in range(n_cols):
        K, K_dB = compute_rician_k_factor_surface_roi(
            H_spec[:, j], H_diff[:, j],
            nadir_lines[j],
            window=SIM['k_factor_window']
        )
        K_map[j] = K_dB

    # 过滤掉无效值
    valid_K = K_map[np.isfinite(K_map)]
    if len(valid_K) > 0:
        K_mean_dB = np.median(valid_K)
    else:
        K_mean_dB = -np.inf

    print(f"  平均K因子(Surface ROI): {K_mean_dB:.1f} dB")

    results = {
        'clutter': clutter,
        'H_spec': H_spec,
        'H_diff': H_diff,
        'K_map_dB': K_map,
        'K_mean_dB': K_mean_dB,
        'nadir_lines': nadir_lines
    }

    return results


# ===========================
# 分层评估指标（补丁3）
# ===========================

def compute_layered_metrics(sim_data, official_data, nadir_lines,
                            peak_window=7, near_window=30):
    """
    分层计算评估指标：峰沿/近峰/背景
    """
    n_rows, n_cols = sim_data.shape

    # 创建层掩码
    peak_mask = np.zeros_like(sim_data, dtype=bool)
    near_mask = np.zeros_like(sim_data, dtype=bool)

    for j in range(n_cols):
        if j < len(nadir_lines):
            nadir = int(nadir_lines[j])

            # 峰沿层
            peak_start = max(0, nadir - peak_window)
            peak_end = min(n_rows, nadir + peak_window)
            peak_mask[peak_start:peak_end, j] = True

            # 近峰层
            near_start = max(0, nadir - near_window)
            near_end = min(n_rows, nadir + near_window)
            near_mask[near_start:near_end, j] = True

    # 背景层
    background_mask = ~near_mask

    # 转换到dB域
    sim_db = 10 * np.log10(np.abs(sim_data) + 1e-12)
    off_db = 10 * np.log10(np.abs(official_data) + 1e-12)

    results = {}

    # 计算各层指标
    for layer_name, mask in [('peak', peak_mask),
                             ('near', near_mask),
                             ('background', background_mask)]:

        valid = mask & (sim_db > -150) & (off_db > -150) & \
                np.isfinite(sim_db) & np.isfinite(off_db)

        if np.sum(valid) < 10:
            results[layer_name] = {
                'rmse_db': np.nan,
                'corr': 0.0,
                'n_pixels': np.sum(valid)
            }
            continue

        sim_valid = sim_db[valid]
        off_valid = off_db[valid]

        # RMSE (dB域)
        rmse_db = np.sqrt(np.mean((sim_valid - off_valid) ** 2))

        # 相关系数
        if len(sim_valid) > 1:
            try:
                corr = np.corrcoef(sim_valid, off_valid)[0, 1]
                if not np.isfinite(corr):
                    corr = 0.0
            except:
                corr = 0.0
        else:
            corr = 0.0

        results[layer_name] = {
            'rmse_db': rmse_db,
            'corr': corr,
            'n_pixels': np.sum(valid)
        }

    return results


# ===========================
# ROI-Gate加权评估（补丁1）
# ===========================

def compute_roi_weighted_metrics(sim_data, official_data, percentile=85):
    """
    ROI-Gate评估：集中在物理有效像素
    """
    # 转换到dB域
    sim_db = 10 * np.log10(np.abs(sim_data) + 1e-12)
    off_db = 10 * np.log10(np.abs(official_data) + 1e-12)

    # ROI阈值
    threshold = np.percentile(off_db[np.isfinite(off_db)], percentile)

    # ROI掩码
    roi = (off_db >= threshold) | (sim_db >= threshold)

    # 加权（强回波权重更高）
    weights = roi.astype(float) * np.maximum(off_db - threshold + 1, 0)
    weights = weights / np.sum(weights)

    # 有效掩码
    valid = roi & (sim_db > -150) & (off_db > -150) & \
            np.isfinite(sim_db) & np.isfinite(off_db)

    if np.sum(valid) < 100:
        return {
            'rmse_db': 100.0,
            'mae_db': 100.0,
            'corr': 0.0,
            'spearman': 0.0,
            'n_pixels': np.sum(valid)
        }

    sim_valid = sim_db[valid]
    off_valid = off_db[valid]
    weights_valid = weights[valid]

    # 加权RMSE
    diff = sim_valid - off_valid
    rmse_db = np.sqrt(np.sum(weights_valid * diff ** 2) / np.sum(weights_valid))
    mae_db = np.sum(weights_valid * np.abs(diff)) / np.sum(weights_valid)

    # 相关系数
    try:
        corr = np.corrcoef(sim_valid, off_valid)[0, 1]
        spearman_corr, _ = spearmanr(sim_valid, off_valid)
    except:
        corr = 0.0
        spearman_corr = 0.0

    return {
        'rmse_db': rmse_db,
        'mae_db': mae_db,
        'corr': corr if np.isfinite(corr) else 0.0,
        'spearman': spearman_corr if np.isfinite(spearman_corr) else 0.0,
        'n_pixels': np.sum(valid),
        'threshold': threshold
    }


# ===========================
# 固定标定评估（补丁2）
# ===========================

def compute_metrics_fixed_calibration(sim_data, official_data, alpha=1.0, beta=None):
    """
    固定斜率α=1，只估偏置β
    """
    # 转换到dB域
    sim_db = 10 * np.log10(np.abs(sim_data) + 1e-12)
    off_db = 10 * np.log10(np.abs(official_data) + 1e-12)

    # 有效掩码
    valid = (sim_db > -150) & (off_db > -150) & \
            np.isfinite(sim_db) & np.isfinite(off_db)

    if np.sum(valid) < 100:
        return {'alpha': 1.0, 'beta': 0.0}

    sim_valid = sim_db[valid]
    off_valid = off_db[valid]

    # 固定α=1，只估β
    if beta is None:
        beta = np.mean(off_valid - alpha * sim_valid)

    return {'alpha': alpha, 'beta': beta}


# ===========================
# 参数可识别性分析V3.1
# ===========================

def parameter_identifiability_analysis_v3(official_sim, geom_df, dt, param_name='epsilon_r'):
    """
    参数可识别性分析V3.1：使用分层评估和ROI权重
    """
    print(f"\n分析参数可识别性(V3.1): {param_name}")

    # 参数扫描范围
    if param_name == 'epsilon_r':
        param_range = np.linspace(2.0, 8.0, SIM['param_steps'])
        baseline_value = 4.0
    elif param_name == 'sigma_h':
        param_range = np.linspace(0.01, 0.2, SIM['param_steps'])
        baseline_value = 0.05
    elif param_name == 'tau_vis':
        param_range = np.linspace(0.1, 2.0, SIM['param_steps'])
        baseline_value = 0.5
    else:
        raise ValueError(f"未知参数: {param_name}")

    print(f"  扫描{len(param_range)}个参数值")
    n_rows, n_cols = official_sim.shape

    # 获取基线标定（α=1固定）
    print(f"  计算基线偏置 ({param_name}={baseline_value:.3f})...")
    env_baseline = EnvironmentalParameters()

    if param_name == 'epsilon_r':
        env_baseline.epsilon_r = baseline_value
        SURFACE_TYPES['basalt']['epsilon_r'] = baseline_value * 1.5
        SURFACE_TYPES['aeolian']['epsilon_r'] = baseline_value * 0.6
    elif param_name == 'sigma_h':
        env_baseline.sigma_h = baseline_value
        SURFACE_TYPES['basalt']['sigma_h'] = baseline_value * 0.4
        SURFACE_TYPES['aeolian']['sigma_h'] = baseline_value * 2.0
    elif param_name == 'tau_vis':
        env_baseline.tau_vis = baseline_value

    # 基线仿真
    baseline_results = forward_simulation_v3(geom_df, n_rows, n_cols, dt, env_baseline)
    baseline_clutter = gaussian_filter(baseline_results['clutter'], sigma=(2.5, 1.0))

    # 固定α=1，只估β
    calib = compute_metrics_fixed_calibration(baseline_clutter, official_sim, alpha=1.0)
    fixed_beta = calib['beta']
    print(f"  固定标定: α=1.0, β={fixed_beta:.1f} dB")

    # 参数扫描
    print(f"  开始参数扫描...")

    # 存储各层结果
    results_by_layer = {
        'peak': {'rmse': [], 'corr': []},
        'near': {'rmse': [], 'corr': []},
        'background': {'rmse': [], 'corr': []},
        'roi': {'rmse': [], 'corr': [], 'spearman': []},
        'global': {'rmse': [], 'corr': []}
    }

    k_factors = []

    for i, param_val in enumerate(param_range):
        print(f"  [{i + 1}/{len(param_range)}] {param_name} = {param_val:.3f}", end=' ')

        # 设置环境参数
        env_params = EnvironmentalParameters()

        if param_name == 'epsilon_r':
            env_params.epsilon_r = param_val
            SURFACE_TYPES['basalt']['epsilon_r'] = param_val * 1.5
            SURFACE_TYPES['aeolian']['epsilon_r'] = param_val * 0.6
        elif param_name == 'sigma_h':
            env_params.sigma_h = param_val
            SURFACE_TYPES['basalt']['sigma_h'] = param_val * 0.4
            SURFACE_TYPES['aeolian']['sigma_h'] = param_val * 2.0
        elif param_name == 'tau_vis':
            env_params.tau_vis = param_val

        # 运行仿真
        results = forward_simulation_v3(geom_df, n_rows, n_cols, dt, env_params)
        clutter = gaussian_filter(results['clutter'], sigma=(2.5, 1.0))

        # 固定标定应用
        clutter_cal = 10 ** ((10 * np.log10(clutter + 1e-12) + fixed_beta) / 10)

        # 分层评估
        layered = compute_layered_metrics(clutter_cal, official_sim,
                                          results['nadir_lines'])

        # ROI评估
        roi = compute_roi_weighted_metrics(clutter_cal, official_sim, percentile=85)

        # 全局评估（作为对比）
        sim_db = 10 * np.log10(clutter_cal + 1e-12)
        off_db = 10 * np.log10(official_sim + 1e-12)
        valid = np.isfinite(sim_db) & np.isfinite(off_db)
        if np.sum(valid) > 0:
            global_rmse = np.sqrt(np.mean((sim_db[valid] - off_db[valid]) ** 2))
            try:
                global_corr = np.corrcoef(sim_db[valid], off_db[valid])[0, 1]
            except:
                global_corr = 0.0
        else:
            global_rmse = 100.0
            global_corr = 0.0

        # 存储结果
        results_by_layer['peak']['rmse'].append(layered['peak']['rmse_db'])
        results_by_layer['peak']['corr'].append(layered['peak']['corr'])
        results_by_layer['near']['rmse'].append(layered['near']['rmse_db'])
        results_by_layer['near']['corr'].append(layered['near']['corr'])
        results_by_layer['background']['rmse'].append(layered['background']['rmse_db'])
        results_by_layer['background']['corr'].append(layered['background']['corr'])
        results_by_layer['roi']['rmse'].append(roi['rmse_db'])
        results_by_layer['roi']['corr'].append(roi['corr'])
        results_by_layer['roi']['spearman'].append(roi['spearman'])
        results_by_layer['global']['rmse'].append(global_rmse)
        results_by_layer['global']['corr'].append(global_corr)

        k_factors.append(results['K_mean_dB'])

        print(f"Peak RMSE={layered['peak']['rmse_db']:.1f}, "
              f"ROI Corr={roi['corr']:.3f}, K={results['K_mean_dB']:.1f} dB")

    # 找最优参数（基于ROI指标）
    roi_rmse = np.array(results_by_layer['roi']['rmse'])
    roi_corr = np.array(results_by_layer['roi']['corr'])

    # 标准化并加权
    def standardize(x):
        x = np.array(x)
        if np.std(x) > 0:
            return (x - np.mean(x)) / np.std(x)
        return x * 0

    score = -standardize(roi_rmse) + standardize(roi_corr)
    best_idx = np.argmax(score)
    best_param = param_range[best_idx]

    print(f"\n  最优{param_name}: {best_param:.3f}")
    print(f"  最佳ROI RMSE: {roi_rmse[best_idx]:.1f} dB")
    print(f"  最佳ROI Corr: {roi_corr[best_idx]:.3f}")
    print(f"  最佳Peak RMSE: {results_by_layer['peak']['rmse'][best_idx]:.1f} dB")

    return {
        'param_range': param_range,
        'results_by_layer': results_by_layer,
        'k_factors': k_factors,
        'best_param': best_param,
        'best_idx': best_idx
    }


# ===========================
# 可视化函数V3.1（修正点4：独立色轴）
# ===========================

def create_main_comparison_plots(sim_results, official_sim, save_prefix=""):
    """创建主要对比图（分开保存）"""

    # 应用PSF
    total_power = np.abs(sim_results['H_spec'] + sim_results['H_diff']) ** 2
    total_power_psf = gaussian_filter(total_power, sigma=(2.5, 1.0))

    # 转换到dB域
    official_db = 10 * np.log10(np.abs(official_sim) + 1e-12)
    sim_db = 10 * np.log10(total_power_psf + 1e-12)

    # 统一色轴
    vmin = np.percentile([official_db[np.isfinite(official_db)],
                          sim_db[np.isfinite(sim_db)]], 2)
    vmax = np.percentile([official_db[np.isfinite(official_db)],
                          sim_db[np.isfinite(sim_db)]], 98)

    # 图1：官方vs仿真对比
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(official_db, aspect='auto', cmap='viridis',
                         vmin=vmin, vmax=vmax)
    axes[0].set_title("Official Clutter (dB)")
    axes[0].set_ylabel("Delay bin")
    axes[0].set_xlabel("Along-track column")
    plt.colorbar(im1, ax=axes[0], label="Power (dB)")

    im2 = axes[1].imshow(sim_db, aspect='auto', cmap='viridis',
                         vmin=vmin, vmax=vmax)
    axes[1].set_title("Our Simulation (dB)")
    axes[1].set_ylabel("Delay bin")
    axes[1].set_xlabel("Along-track column")
    plt.colorbar(im2, ax=axes[1], label="Power (dB)")

    diff_db = sim_db - official_db
    diff_max = np.percentile(np.abs(diff_db[np.isfinite(diff_db)]), 98)
    im3 = axes[2].imshow(diff_db, aspect='auto', cmap='RdBu_r',
                         vmin=-diff_max, vmax=diff_max)
    axes[2].set_title("Difference (dB)")
    axes[2].set_ylabel("Delay bin")
    axes[2].set_xlabel("Along-track column")
    plt.colorbar(im3, ax=axes[2], label="Δ(dB)")

    plt.suptitle("Mars ISAC Channel Model Validation - Main Comparison")
    plt.tight_layout()
    filename1 = f"{save_prefix}comparison_main.png"
    plt.savefig(filename1, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {filename1}")
    plt.show()

    # 图2：信道分解（修正点4：独立色轴）
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 镜面分量（独立色轴）
    spec_db = 10 * np.log10(np.abs(sim_results['H_spec']) ** 2 + 1e-12)
    spec_valid = spec_db[np.isfinite(spec_db)]
    spec_vmin = np.percentile(spec_valid, 2)
    spec_vmax = np.percentile(spec_valid, 98)

    im1 = axes[0, 0].imshow(spec_db, aspect='auto', cmap='hot',
                            vmin=spec_vmin, vmax=spec_vmax)
    axes[0, 0].set_title("Specular Component (dB)")
    axes[0, 0].set_ylabel("Delay bin")
    axes[0, 0].set_xlabel("Along-track column")
    plt.colorbar(im1, ax=axes[0, 0], label="Power (dB)")

    # 漫散射分量（独立色轴）
    diff_db = 10 * np.log10(np.abs(sim_results['H_diff']) ** 2 + 1e-12)
    diff_valid = diff_db[np.isfinite(diff_db)]
    diff_vmin = np.percentile(diff_valid, 2)
    diff_vmax = np.percentile(diff_valid, 98)

    im2 = axes[0, 1].imshow(diff_db, aspect='auto', cmap='cool',
                            vmin=diff_vmin, vmax=diff_vmax)
    axes[0, 1].set_title("Diffuse Component (dB)")
    axes[0, 1].set_ylabel("Delay bin")
    axes[0, 1].set_xlabel("Along-track column")
    plt.colorbar(im2, ax=axes[0, 1], label="Power (dB)")

    # K因子分布图（基于表面ROI计算）
    K_map_2d = np.tile(sim_results['K_map_dB'], (100, 1))
    K_valid = sim_results['K_map_dB'][np.isfinite(sim_results['K_map_dB'])]
    if len(K_valid) > 0:
        k_vmin = np.percentile(K_valid, 5)
        k_vmax = np.percentile(K_valid, 95)
    else:
        k_vmin, k_vmax = -20, 10

    im3 = axes[1, 0].imshow(K_map_2d, aspect='auto', cmap='RdYlBu_r',
                            vmin=k_vmin, vmax=k_vmax)
    axes[1, 0].set_title("Rician K-factor Map (Surface ROI, dB)")
    axes[1, 0].set_ylabel("(Extended for visualization)")
    axes[1, 0].set_xlabel("Along-track column")
    plt.colorbar(im3, ax=axes[1, 0], label="K (dB)")

    # 功率剖面
    spec_mean = np.mean(spec_db, axis=0)
    diff_mean = np.mean(diff_db, axis=0)
    axes[1, 1].plot(spec_mean, 'r-', label='Specular', linewidth=1.5, alpha=0.8)
    axes[1, 1].plot(diff_mean, 'b-', label='Diffuse', linewidth=1.5, alpha=0.8)
    axes[1, 1].set_title("Mean Power Profile")
    axes[1, 1].set_xlabel("Along-track column")
    axes[1, 1].set_ylabel("Power (dB)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 添加K因子统计信息
    axes[1, 1].text(0.02, 0.98, f"Mean K: {sim_results['K_mean_dB']:.1f} dB",
                    transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Channel Decomposition Analysis")
    plt.tight_layout()
    filename2 = f"{save_prefix}channel_decomposition.png"
    plt.savefig(filename2, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {filename2}")
    plt.show()


def create_parameter_sensitivity_plots(identifiability_results, save_prefix=""):
    """创建参数敏感性分析图（分层评估）"""

    n_params = len(identifiability_results)
    fig, axes = plt.subplots(n_params, 3, figsize=(15, 5 * n_params))

    if n_params == 1:
        axes = axes.reshape(1, -1)

    for idx, (param_name, results) in enumerate(identifiability_results.items()):
        param_range = results['param_range']
        layers = results['results_by_layer']
        k_factors = results['k_factors']

        # RMSE by layer
        ax = axes[idx, 0]
        ax.plot(param_range, layers['peak']['rmse'], 'r-', marker='o',
                label='Peak', linewidth=2)
        ax.plot(param_range, layers['near']['rmse'], 'g-', marker='^',
                label='Near-peak', linewidth=1.5)
        ax.plot(param_range, layers['background']['rmse'], 'b-', marker='s',
                label='Background', linewidth=1, alpha=0.5)
        ax.plot(param_range, layers['roi']['rmse'], 'k--', marker='*',
                label='ROI(85%)', linewidth=2)
        ax.axvline(x=results['best_param'], color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel(param_name)
        ax.set_ylabel("RMSE (dB)")
        ax.set_title(f"Layered RMSE - {param_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        # Correlation by layer
        ax = axes[idx, 1]
        ax.plot(param_range, layers['peak']['corr'], 'r-', marker='o',
                label='Peak', linewidth=2)
        ax.plot(param_range, layers['near']['corr'], 'g-', marker='^',
                label='Near-peak', linewidth=1.5)
        ax.plot(param_range, layers['roi']['corr'], 'k--', marker='*',
                label='ROI(85%)', linewidth=2)
        if 'spearman' in layers['roi']:
            ax.plot(param_range, layers['roi']['spearman'], 'm:', marker='d',
                    label='ROI Spearman', linewidth=1.5)
        ax.axvline(x=results['best_param'], color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel(param_name)
        ax.set_ylabel("Correlation")
        ax.set_title(f"Layered Correlation - {param_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # K-factor evolution
        ax = axes[idx, 2]
        ax.plot(param_range, k_factors, 'b-', marker='o', linewidth=2)
        ax.axvline(x=results['best_param'], color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel(param_name)
        ax.set_ylabel("K-factor (dB)")
        ax.set_title(f"Rician K-factor (Surface ROI) vs {param_name}")
        ax.grid(True, alpha=0.3)

        # 添加文本注释
        best_idx = results['best_idx']
        text = f"Optimal: {results['best_param']:.3f}\n"
        text += f"Peak RMSE: {layers['peak']['rmse'][best_idx]:.1f} dB\n"
        text += f"ROI Corr: {layers['roi']['corr'][best_idx]:.3f}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Parameter Sensitivity Analysis - Layered Evaluation", fontsize=14)
    plt.tight_layout()
    filename = f"{save_prefix}parameter_sensitivity_layered.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {filename}")
    plt.show()


def create_evaluation_metrics_plot(sim_results, official_sim, save_prefix=""):
    """创建评估指标汇总图"""

    total_power = np.abs(sim_results['H_spec'] + sim_results['H_diff']) ** 2
    total_power_psf = gaussian_filter(total_power, sigma=(2.5, 1.0))

    # 计算各种评估
    layered = compute_layered_metrics(total_power_psf, official_sim,
                                      sim_results['nadir_lines'])
    roi = compute_roi_weighted_metrics(total_power_psf, official_sim, percentile=85)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 分层RMSE对比
    ax = axes[0, 0]
    layers = ['Peak', 'Near-peak', 'Background', 'ROI(85%)']
    rmse_values = [layered['peak']['rmse_db'],
                   layered['near']['rmse_db'],
                   layered['background']['rmse_db'],
                   roi['rmse_db']]
    colors = ['red', 'green', 'blue', 'black']
    bars = ax.bar(layers, rmse_values, color=colors, alpha=0.7)
    ax.set_ylabel('RMSE (dB)')
    ax.set_title('Layered RMSE Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmse_values):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom')

    # 分层相关系数对比
    ax = axes[0, 1]
    corr_values = [layered['peak']['corr'],
                   layered['near']['corr'],
                   layered['background']['corr'],
                   roi['corr']]
    bars = ax.bar(layers, corr_values, color=colors, alpha=0.7)
    ax.set_ylabel('Correlation')
    ax.set_title('Layered Correlation Comparison')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, corr_values):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')

    # ROI散点图
    ax = axes[1, 0]
    official_db = 10 * np.log10(np.abs(official_sim) + 1e-12)
    sim_db = 10 * np.log10(total_power_psf + 1e-12)

    threshold = roi['threshold']
    roi_mask = (official_db >= threshold) | (sim_db >= threshold)
    sample_mask = roi_mask & (np.random.rand(*sim_db.shape) < 0.1)

    if np.sum(sample_mask) > 0:
        ax.scatter(official_db[sample_mask], sim_db[sample_mask],
                   alpha=0.3, s=1, c='blue')

        # 拟合线
        valid = np.isfinite(official_db[sample_mask]) & np.isfinite(sim_db[sample_mask])
        if np.sum(valid) > 10:
            z = np.polyfit(official_db[sample_mask][valid],
                           sim_db[sample_mask][valid], 1)
            p = np.poly1d(z)
            x_fit = np.linspace(threshold, np.max(official_db[sample_mask][valid]), 100)
            ax.plot(x_fit, p(x_fit), 'g-', alpha=0.5,
                    label=f'Fit: y={z[0]:.2f}x+{z[1]:.1f}')

    ax.plot([threshold, threshold + 30], [threshold, threshold + 30], 'r--', label='1:1 line')
    ax.set_xlabel('Official (dB)')
    ax.set_ylabel('Simulation (dB)')
    ax.set_title(f'ROI Scatter (≥{threshold:.1f} dB)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 指标汇总表
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    === Evaluation Metrics Summary ===

    ROI Evaluation (85th percentile):
      RMSE: {roi['rmse_db']:.1f} dB
      MAE: {roi['mae_db']:.1f} dB
      Pearson: {roi['corr']:.3f}
      Spearman: {roi['spearman']:.3f}
      Pixels: {roi['n_pixels']}

    Peak Layer:
      RMSE: {layered['peak']['rmse_db']:.1f} dB
      Correlation: {layered['peak']['corr']:.3f}
      Pixels: {layered['peak']['n_pixels']}

    K-factor (Surface ROI): {sim_results['K_mean_dB']:.1f} dB

    Fixed Calibration: α=1.0, β estimated
    """

    ax.text(0.1, 0.9, summary_text.strip(), transform=ax.transAxes,
            fontsize=11, family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle("Evaluation Metrics Analysis", fontsize=14)
    plt.tight_layout()
    filename = f"{save_prefix}evaluation_metrics.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {filename}")
    plt.show()


def create_metrics_summary_table(identifiability_results):
    """创建参数映射表"""
    print("\n" + "=" * 80)
    print("Environmental Parameter Mapping - V3.1 Results")
    print("=" * 80)

    headers = ["Parameter", "Optimal", "Peak RMSE", "Peak Corr",
               "ROI RMSE", "ROI Corr", "K-factor"]
    rows = []

    for param_name, results in identifiability_results.items():
        best_idx = results['best_idx']
        layers = results['results_by_layer']

        row = [
            param_name,
            f"{results['best_param']:.3f}",
            f"{layers['peak']['rmse'][best_idx]:.1f}",
            f"{layers['peak']['corr'][best_idx]:.3f}",
            f"{layers['roi']['rmse'][best_idx]:.1f}",
            f"{layers['roi']['corr'][best_idx]:.3f}",
            f"{results['k_factors'][best_idx]:.1f}"
        ]
        rows.append(row)

    # 打印表格
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2
                  for i in range(len(headers))]

    header_line = "|".join(h.center(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        data_line = "|".join(str(v).center(w) for v, w in zip(row, col_widths))
        print(data_line)

    print("=" * 80)

    # 保存CSV
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv("parameter_mapping_v3_1.csv", index=False)
    print("✓ 参数映射表已保存: parameter_mapping_v3_1.csv")


# ===========================
# 数据加载函数
# ===========================

def load_sharad_data(track_id='s_00810101'):
    """加载SHARAD数据"""
    from pds4_tools import pds4_read

    # 官方杂波
    sim_xml = os.path.join(DATA_DIR['clutter'], f"{track_id}_sim.xml")
    print(f"读取官方杂波: {sim_xml}")
    prod = pds4_read(sim_xml)

    if len(prod) > 1:
        official_sim = prod[-1].data
    else:
        official_sim = prod[0].data

    print(f"  杂波尺寸: {official_sim.shape}")

    # 几何数据
    rtrn_csv = os.path.join(DATA_DIR['clutter'], f"{track_id}_rtrn.csv")
    geom_df = pd.read_csv(rtrn_csv)
    print(f"  几何数据: {len(geom_df)} 行")

    # 采样间隔
    rgram_xml = os.path.join(DATA_DIR['radargram'], f"{track_id}_rgram.xml")
    dt = 0.375e-6  # 默认值

    if os.path.exists(rgram_xml):
        try:
            rg = pds4_read(rgram_xml)
            for arr in rg:
                if hasattr(arr, 'meta_data'):
                    for key in ['line_sampling_interval', 'Line_Sampling_Interval']:
                        if key in arr.meta_data:
                            dt = float(arr.meta_data[key])
                            print(f"  采样间隔: {dt * 1e6:.3f} μs")
                            break
        except:
            pass

    return official_sim, geom_df, dt


# ===========================
# 主验证流程
# ===========================

def main():
    """完整的理论模型验证流程 V3.1"""

    print("=" * 80)
    print("火星ISAC信道模型验证 V3.1 - 修正版")
    print("对应论文: Section II - The Unified Mars ISAC Channel Model")
    print("=" * 80)

    print("\n主要修正：")
    print("1. K因子基于表面ROI计算（±25 bins窗口）")
    print("2. 大气衰减功率/幅度口径统一")
    print("3. 坡度计算加入cos(φ)纬度修正")
    print("4. 信道分解图使用独立色轴")

    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    load_mola()

    if len(MOLA_TILES) == 0:
        print("错误: 无MOLA数据")
        return

    official_sim, geom_df, dt = load_sharad_data()
    n_rows, n_cols = official_sim.shape

    # 2. 设置环境参数
    print("\n[步骤2] 初始化环境参数...")
    env_params = EnvironmentalParameters()
    env_params.k0 = 0.007
    env_params.n_lambert = 1.5

    # 设置地表类型参数
    SURFACE_TYPES['basalt']['k0'] = env_params.k0 * 0.6
    SURFACE_TYPES['aeolian']['k0'] = env_params.k0 * 1.6

    print(f"\n环境参数:")
    print(f"  τ_vis = {env_params.tau_vis}")
    print(f"  ε_r = {env_params.epsilon_r}")
    print(f"  σ_h = {env_params.sigma_h} m")
    print(f"  k0 = {env_params.k0}")
    print(f"  n_lambert = {env_params.n_lambert}")

    # 3. 理论模型前向仿真
    print("\n[步骤3] 执行理论模型前向仿真...")
    sim_results = forward_simulation_v3(geom_df, n_rows, n_cols, dt, env_params)

    # 4. 创建主要对比图
    print("\n[步骤4] 生成主要对比图...")
    create_main_comparison_plots(sim_results, official_sim, save_prefix="v3_1_")

    # 5. 参数可识别性分析
    print("\n[步骤5] 参数可识别性分析...")
    identifiability_results = {}

    for param in ['epsilon_r', 'sigma_h']:
        ident = parameter_identifiability_analysis_v3(
            official_sim, geom_df, dt, param)
        identifiability_results[param] = ident

    # 6. 生成参数敏感性图
    print("\n[步骤6] 生成参数敏感性分析图...")
    create_parameter_sensitivity_plots(identifiability_results, save_prefix="v3_1_")

    # 7. 生成评估指标图
    print("\n[步骤7] 生成评估指标汇总图...")
    create_evaluation_metrics_plot(sim_results, official_sim, save_prefix="v3_1_")

    # 8. 生成参数映射表
    print("\n[步骤8] 生成参数映射表...")
    create_metrics_summary_table(identifiability_results)

    # 9. 最终评估（可选：全分辨率）
    if FULL_RES_EVAL and SIM['col_subsample'] > 1:
        print("\n[步骤9] 全分辨率最终评估...")
        print("  运行全分辨率仿真以获得精确指标...")

        # 使用最优参数
        env_optimal = EnvironmentalParameters()
        for param_name, results in identifiability_results.items():
            if param_name == 'epsilon_r':
                env_optimal.epsilon_r = results['best_param']
            elif param_name == 'sigma_h':
                env_optimal.sigma_h = results['best_param']

        sim_results_full = forward_simulation_v3(geom_df, n_rows, n_cols, dt,
                                                 env_optimal, full_resolution=True)

        # 计算最终指标
        total_power_full = np.abs(sim_results_full['H_spec'] + sim_results_full['H_diff']) ** 2
        total_power_psf_full = gaussian_filter(total_power_full, sigma=(2.5, 1.0))

        layered_full = compute_layered_metrics(total_power_psf_full, official_sim,
                                               sim_results_full['nadir_lines'])
        roi_full = compute_roi_weighted_metrics(total_power_psf_full, official_sim)

        print("\n全分辨率最终结果:")
        print(f"  Peak RMSE: {layered_full['peak']['rmse_db']:.1f} dB")
        print(f"  Peak Corr: {layered_full['peak']['corr']:.3f}")
        print(f"  ROI RMSE: {roi_full['rmse_db']:.1f} dB")
        print(f"  ROI Corr: {roi_full['corr']:.3f}")
        print(f"  K-factor: {sim_results_full['K_mean_dB']:.1f} dB")

    print("\n" + "=" * 80)
    print("✓ 验证完成！")
    print("=" * 80)

    print("\n生成的文件:")
    print("  - v3_1_comparison_main.png (主要对比)")
    print("  - v3_1_channel_decomposition.png (信道分解)")
    print("  - v3_1_parameter_sensitivity_layered.png (参数敏感性)")
    print("  - v3_1_evaluation_metrics.png (评估指标)")
    print("  - parameter_mapping_v3_1.csv (参数映射表)")

    return {
        'sim_results': sim_results,
        'identifiability': identifiability_results
    }


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()