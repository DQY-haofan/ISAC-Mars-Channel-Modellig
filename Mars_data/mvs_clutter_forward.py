#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
火星ISAC信道模型完整验证框架
整合理论模型、MVS前向仿真、参数识别性验证
对应论文Section II: The Unified Mars ISAC Channel Model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.ndimage import sobel, uniform_filter
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import time
import warnings

warnings.filterwarnings('ignore')


# ===========================
# 替代的图像质量评估指标（无需scikit-image）
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
# 物理常量（对应论文符号）
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
# 仿真质量控制（速度vs精度权衡）
# ===========================
FAST_MODE = True  # 快速模式开关

if FAST_MODE:
    SIM = {
        "col_subsample": 20,  # 列下采样（20倍加速）
        "patch_size": 21,  # patch大小（2.4倍加速）
        "patch_km": 10.0,  # patch范围km
        "gate_bins": 300,  # NadirLine附近的有效窗口
        "param_steps": 7  # 参数扫描步数
    }
else:
    SIM = {
        "col_subsample": 5,  # 原始设置
        "patch_size": 51,
        "patch_km": 25.0,
        "gate_bins": 600,
        "param_steps": 15
    }


# ===========================
# 环境参数向量 θ_env（论文Table I）
# ===========================
class EnvironmentalParameters:
    """统一环境状态向量 θ_env"""

    def __init__(self):
        self.tau_vis = 0.5  # 尘埃光学深度
        self.epsilon_r = 4.0  # 表面复介电常数（实部）
        self.sigma_h = 0.05  # 表面RMS高度 (m)
        self.correlation_length = 1.0  # 相关长度 (m)
        self.S4 = 0.3  # 电离层闪烁指数
        self.Cn2_surface = 1e-14  # 地表折射率结构常数

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
# 地表类型定义（论文中的两类）
# ===========================
SURFACE_TYPES = {
    'basalt': {
        'epsilon_r': 6.0,
        'sigma_h': 0.02,
        'correlation_length': 0.5,
        'w_spec': 0.8,  # 镜面权重
        'k_d': 0.1  # 漫反射系数
    },
    'aeolian': {
        'epsilon_r': 2.5,
        'sigma_h': 0.10,
        'correlation_length': 2.0,
        'w_spec': 0.2,
        'k_d': 0.15
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
# 理论模型实现（论文Section II.C）
# ===========================

def fresnel_reflection_coefficient(theta_i, epsilon_r, polarization='H'):
    """
    计算Fresnel反射系数（论文Eq.14前半部分）

    Args:
        theta_i: 入射角 (rad)
        epsilon_r: 复介电常数
        polarization: 'H'水平极化, 'V'垂直极化
    """
    cos_theta = np.cos(theta_i)
    sin_theta = np.sin(theta_i)

    # 避免复数问题
    sqrt_term = np.sqrt(np.maximum(0, epsilon_r - sin_theta ** 2))

    if polarization == 'H':
        # 水平极化
        Gamma = (cos_theta - sqrt_term) / (cos_theta + sqrt_term)
    else:
        # 垂直极化
        Gamma = (epsilon_r * cos_theta - sqrt_term) / (epsilon_r * cos_theta + sqrt_term)

    return np.abs(Gamma) ** 2  # 功率反射率


def effective_reflection_coefficient(theta_i, epsilon_r, sigma_h, lambda_=LAMBDA):
    """
    有效反射系数，包含粗糙度效应（论文Eq.14）
    Γ_eff = Γ(θ,ε_r) × exp(-(4π·σ_h·cosθ/λ)²)
    """
    # Fresnel系数
    Gamma = fresnel_reflection_coefficient(theta_i, epsilon_r)

    # 粗糙度衰减因子
    roughness_factor = np.exp(-(4 * np.pi * sigma_h * np.cos(theta_i) / lambda_) ** 2)

    return Gamma * roughness_factor


def select_scattering_model(k, sigma_h, correlation_length):
    """
    混合散射模型选择（论文Eq.16）
    根据有效性准则自适应选择KA或SPM
    """
    kl = k * correlation_length
    k_sigma = k * sigma_h

    # 有效性准则
    if kl > 6 and correlation_length ** 2 > 2.76 * sigma_h * LAMBDA:
        return 'KA'  # Kirchhoff近似
    elif k_sigma < 0.3:
        return 'SPM'  # 小扰动方法
    else:
        return 'IEM'  # 中间情况用IEM


def compute_diffuse_scattering(theta_i, sigma_h, correlation_length, epsilon_r):
    """
    计算漫散射功率（基于选定的模型）
    """
    model = select_scattering_model(K, sigma_h, correlation_length)

    if model == 'SPM':
        # 小扰动方法
        cos_theta = np.cos(theta_i)
        sin_theta = np.sin(theta_i)

        # SPM散射系数
        sigma0_diff = (K ** 4 * sigma_h ** 2 * correlation_length ** 2 / np.pi) * \
                      np.exp(-(K * correlation_length * sin_theta) ** 2) * \
                      np.abs((epsilon_r - 1) / (epsilon_r + 1)) ** 2 * cos_theta ** 4

    elif model == 'KA':
        # Kirchhoff近似（几何光学）
        sigma0_diff = 0.1 * np.cos(theta_i)  # 简化的漫反射

    else:  # IEM
        # 积分方程方法（简化版）
        sigma0_diff = 0.05 * np.cos(theta_i)

    return sigma0_diff


def compute_rician_k_factor(specular_power, diffuse_power):
    """
    计算Rician K因子（论文Eq.17）
    K = |H_Spec|² / E[|H_Diff|²]
    """
    K = specular_power / (diffuse_power + 1e-10)
    K_dB = 10 * np.log10(K + 1e-10)
    return K, K_dB


def atmospheric_attenuation(distance, tau_vis, frequency=FC):
    """
    大气衰减模型（论文Eq.10）
    包含尘埃消光效应
    """
    # 简化的衰减模型 (dB/km)
    A_dust = 0.1 * tau_vis * (frequency / 1e9) ** 0.5  # 尘埃衰减

    # 总路径损耗 (dB)
    L_atm = A_dust * (distance / 1000.0)

    return 10 ** (-L_atm / 20.0)  # 线性衰减因子


def compute_channel_coherence(scattering_function):
    """
    计算信道相干参数（论文Eq.8）
    T_c = [∫∫ν²S_H(τ,ν)dτdν]^(-1/2)
    B_c = [∫∫τ²S_H(τ,ν)dτdν]^(-1/2)
    """
    # 简化计算
    delay_spread = np.std(scattering_function, axis=0)
    doppler_spread = np.std(scattering_function, axis=1)

    T_c = 1.0 / (2 * np.pi * np.mean(doppler_spread))
    B_c = 1.0 / (2 * np.pi * np.mean(delay_spread))

    return T_c, B_c


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


# 全局MOLA存储
MOLA_TILES = []


def load_mola():
    """加载所有MOLA瓦片"""
    global MOLA_TILES

    tiles = [
        ('megt00n090hb.img', 'megt00n090hb.lbl'),
        ('megt44s090hb.img', 'megt44s090hb.lbl')
    ]

    print(f"MOLA数据目录: {DATA_DIR['mola']}")
    print(f"当前工作目录: {os.getcwd()}")

    for img_file, lbl_file in tiles:
        img_path = os.path.join(DATA_DIR['mola'], img_file)
        lbl_path = os.path.join(DATA_DIR['mola'], lbl_file)

        print(f"\n检查文件:")
        print(f"  IMG: {img_path} - 存在: {os.path.exists(img_path)}")
        print(f"  LBL: {lbl_path} - 存在: {os.path.exists(lbl_path)}")

        if os.path.exists(img_path) and os.path.exists(lbl_path):
            try:
                tile = read_mola_tile(img_path, lbl_path)
                MOLA_TILES.append(tile)
                print(f"  ✓ 成功加载: {img_file}")
            except Exception as e:
                print(f"  ✗ 加载失败: {e}")
        else:
            if not os.path.exists(img_path):
                print(f"  ✗ 缺少IMG文件: {img_path}")
            if not os.path.exists(lbl_path):
                print(f"  ✗ 缺少LBL文件: {lbl_path}")

    if len(MOLA_TILES) == 0:
        print("\n可能的解决方案:")
        print("1. 确保MOLA文件在 'MOLA MEGDR' 文件夹中")
        print("2. 检查文件名是否正确:")
        print("   - megt00n090hb.img 和 megt00n090hb.lbl")
        print("   - megt44s090hb.img 和 megt44s090hb.lbl")
        print("3. 或者修改脚本中的 DATA_DIR['mola'] 路径")
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

            # 双线性插值
            i0, j0 = int(i), int(j)
            di, dj = i - i0, j - j0

            v00 = tile['data'][i0, j0]
            v01 = tile['data'][i0, j0 + 1]
            v10 = tile['data'][i0 + 1, j0]
            v11 = tile['data'][i0 + 1, j0 + 1]

            if np.isnan([v00, v01, v10, v11]).any():
                return np.nan

            return (1 - di) * (1 - dj) * v00 + (1 - di) * dj * v01 + di * (1 - dj) * v10 + di * dj * v11

    return np.nan


# ===========================
# 增强版MVS前向仿真器
# ===========================

def forward_simulation_enhanced(geom_df, n_rows, n_cols, dt, env_params):
    """
    增强版前向仿真，使用理论模型
    对应论文Eq.(1): H = H_LoS + H_Spec + H_Targ + H_Diff
    """
    print("\n执行理论模型前向仿真...")
    print(f"  模式: {'快速' if FAST_MODE else '高精度'}")
    print(f"  列下采样: 1/{SIM['col_subsample']}")
    print(f"  Patch: {SIM['patch_size']}×{SIM['patch_size']} ({SIM['patch_km']}×{SIM['patch_km']} km)")

    # 初始化信道分量
    H_spec = np.zeros((n_rows, n_cols), dtype=complex)
    H_diff = np.zeros((n_rows, n_cols), dtype=complex)

    # 用于K因子计算
    spec_power_map = np.zeros(n_cols)
    diff_power_map = np.zeros(n_cols)

    # 列下采样
    subsample = SIM['col_subsample']
    cols_to_process = np.arange(0, min(n_cols, len(geom_df)), subsample)

    # 统计
    valid_facets = 0
    total_spec_power = 0
    total_diff_power = 0

    for idx, j in enumerate(cols_to_process):
        if idx % 50 == 0:
            print(f"  进度: {idx}/{len(cols_to_process)}", end='\r')

        # 获取几何信息，添加默认值防止NaN
        try:
            sc_lon = float(geom_df.iloc[j].get('SpacecraftLon', 140))
            sc_lat = float(geom_df.iloc[j].get('SpacecraftLat', -45))
            sc_hgt = float(geom_df.iloc[j].get('SpacecraftHgt', 3000))

            gnd_lon = float(geom_df.iloc[j].get('FirstLon', sc_lon))
            gnd_lat = float(geom_df.iloc[j].get('FirstLat', sc_lat))

            # NadirLine用于对齐
            nadir_line = float(geom_df.iloc[j].get('NadirLine', 1800))
            nadir_hgt = float(geom_df.iloc[j].get('NadirHgt', 0))

            # 确保值合理
            if np.isnan(sc_hgt) or sc_hgt < 100 or sc_hgt > 10000:
                sc_hgt = 3000
            if np.isnan(nadir_line):
                nadir_line = 1800  # 默认中间位置

        except Exception as e:
            continue

        if sc_lon < 0: sc_lon += 360
        if gnd_lon < 0: gnd_lon += 360

        # 计算名义两程距离（用于对齐）
        R_nadir = sc_hgt - nadir_hgt
        tau_nadir = 2 * R_nadir / C0

        # 采样局部DEM
        patch_size = SIM['patch_size']
        dlat = SIM['patch_km'] / 111.0
        dlon = dlat / max(0.001, np.cos(gnd_lat * DEG))

        lats = np.linspace(gnd_lat - dlat / 2, gnd_lat + dlat / 2, patch_size)
        lons = np.linspace(gnd_lon - dlon / 2, gnd_lon + dlon / 2, patch_size)

        col_spec_power = 0
        col_diff_power = 0

        # 处理patch（跳跃采样进一步加速）
        step = 2 if FAST_MODE else 1
        for ii in range(0, patch_size, step):
            for jj in range(0, patch_size, step):
                facet_lat = lats[ii]
                facet_lon = lons[jj]
                facet_elev = sample_mola(facet_lat, facet_lon)

                if np.isnan(facet_elev):
                    facet_elev = 0  # 使用默认高度

                # 简化几何计算
                # 使用球面近似计算斜距
                dlat_rad = (facet_lat - sc_lat) * DEG
                dlon_rad = (facet_lon - sc_lon) * DEG * np.cos(sc_lat * DEG)

                # 水平距离
                d_horiz = MARS_R * np.sqrt(dlat_rad ** 2 + dlon_rad ** 2)

                # 垂直距离
                d_vert = (sc_hgt - facet_elev)

                # 斜距
                R = np.sqrt(d_horiz ** 2 + d_vert ** 2)

                # 确保R合理
                if R < 100 or R > 10000:
                    continue

                # 入射角（简化）
                theta_i = np.arctan(d_horiz / max(100, d_vert))
                theta_i = np.clip(theta_i, 0, 80 * DEG)

                # 选择地表类型
                if abs(facet_lat) < 30:
                    surface = SURFACE_TYPES['basalt']
                else:
                    surface = SURFACE_TYPES['aeolian']

                # 计算镜面反射
                Gamma_eff = effective_reflection_coefficient(
                    theta_i, surface['epsilon_r'], surface['sigma_h'])

                # 计算漫散射
                sigma0_diff = compute_diffuse_scattering(
                    theta_i, surface['sigma_h'],
                    1.0, surface['epsilon_r'])  # correlation_length=1.0

                # 大气衰减
                L_atm = atmospheric_attenuation(R, env_params.tau_vis)

                # 面元面积
                facet_area = (SIM['patch_km'] * 1000 / patch_size) ** 2

                # 功率计算
                P_spec = (LAMBDA ** 2 / (4 * np.pi) ** 3) * (1.0 / R ** 4) * \
                         Gamma_eff * facet_area * L_atm

                P_diff = (LAMBDA ** 2 / (4 * np.pi) ** 3) * (1.0 / R ** 4) * \
                         sigma0_diff * facet_area * L_atm

                # 时延计算（关键修正：使用NadirLine对齐）
                tau = 2 * R / C0
                dtau = tau - tau_nadir  # 相对于nadir的时延差
                dk = int(round(dtau / dt))  # 相对bin偏移

                # 门控：只在NadirLine附近累加
                if abs(dk) > SIM['gate_bins']:
                    continue

                k = int(nadir_line) + dk  # 实际bin位置

                if 0 <= k < n_rows:
                    # 波束权重（可选）
                    beam_weight = np.exp(-(dk / (SIM['gate_bins'] / 3)) ** 2)

                    # 添加相位
                    phase_spec = np.exp(-1j * 2 * np.pi * FC * tau)
                    phase_diff = np.exp(-1j * 2 * np.pi * FC * tau) * \
                                 np.exp(1j * 2 * np.pi * np.random.rand())

                    H_spec[k, j] += np.sqrt(P_spec) * phase_spec * beam_weight
                    H_diff[k, j] += np.sqrt(P_diff) * phase_diff * beam_weight

                    col_spec_power += P_spec
                    col_diff_power += P_diff
                    valid_facets += 1

        # 记录功率
        spec_power_map[j] = col_spec_power
        diff_power_map[j] = col_diff_power
        total_spec_power += col_spec_power
        total_diff_power += col_diff_power

    print(f"\n  有效面元: {valid_facets}")

    # 填充未处理的列（最近邻插值）
    for j in range(n_cols):
        if j not in cols_to_process:
            nearest = cols_to_process[np.argmin(np.abs(cols_to_process - j))]
            H_spec[:, j] = H_spec[:, nearest]
            H_diff[:, j] = H_diff[:, nearest]
            spec_power_map[j] = spec_power_map[nearest]
            diff_power_map[j] = diff_power_map[nearest]

    # 避免全零
    if np.max(np.abs(H_spec)) == 0:
        H_spec += (np.random.randn(n_rows, n_cols) + 1j * np.random.randn(n_rows, n_cols)) * 1e-6
    if np.max(np.abs(H_diff)) == 0:
        H_diff += (np.random.randn(n_rows, n_cols) + 1j * np.random.randn(n_rows, n_cols)) * 1e-6

    # 合成总信道
    H_total = H_spec + H_diff
    clutter = np.abs(H_total) ** 2

    # 计算K因子（避免NaN）
    mean_spec = np.mean(spec_power_map[spec_power_map > 0]) if np.any(spec_power_map > 0) else 1e-10
    mean_diff = np.mean(diff_power_map[diff_power_map > 0]) if np.any(diff_power_map > 0) else 1e-10

    K_mean, K_dB_mean = compute_rician_k_factor(mean_spec, mean_diff)

    print(f"  镜面功率: {total_spec_power:.3e}")
    print(f"  漫散射功率: {total_diff_power:.3e}")
    print(f"  平均K因子: {K_dB_mean:.1f} dB")

    results = {
        'clutter': clutter,
        'H_spec': H_spec,
        'H_diff': H_diff,
        'K_factor_map': spec_power_map / (diff_power_map + 1e-10),
        'K_mean_dB': K_dB_mean
    }

    return results


# ===========================
# 参数可识别性分析（论文Section II.C末尾）
# ===========================

def parameter_identifiability_analysis(official_sim, geom_df, dt, param_name='epsilon_r'):
    """
    验证环境参数的可识别性
    通过参数扫描计算Fisher信息矩阵元素
    """
    print(f"\n分析参数可识别性: {param_name}")

    # 参数扫描范围（基于火星实际范围）
    if param_name == 'epsilon_r':
        param_range = np.linspace(2.0, 8.0, SIM['param_steps'])
    elif param_name == 'sigma_h':
        param_range = np.linspace(0.01, 0.2, SIM['param_steps'])
    elif param_name == 'tau_vis':
        param_range = np.linspace(0.1, 2.0, SIM['param_steps'])
    else:
        raise ValueError(f"未知参数: {param_name}")

    print(f"  扫描{len(param_range)}个参数值")

    # 存储结果
    ssim_values = []
    corr_values = []

    n_rows, n_cols = official_sim.shape

    for i, param_val in enumerate(param_range):
        print(f"  [{i + 1}/{len(param_range)}] {param_name} = {param_val:.3f}", end=' ')

        # 设置环境参数
        env_params = EnvironmentalParameters()
        if param_name == 'epsilon_r':
            env_params.epsilon_r = param_val
            # 更新所有地表类型
            for surface in SURFACE_TYPES.values():
                surface['epsilon_r'] = param_val
        elif param_name == 'sigma_h':
            env_params.sigma_h = param_val
            for surface in SURFACE_TYPES.values():
                surface['sigma_h'] = param_val
        elif param_name == 'tau_vis':
            env_params.tau_vis = param_val

        # 运行仿真
        results = forward_simulation_enhanced(geom_df, n_rows, n_cols, dt, env_params)
        clutter = results['clutter']

        # 计算相似度
        clutter_norm = clutter / (np.max(clutter) + 1e-10)
        official_norm = np.abs(official_sim) / (np.max(np.abs(official_sim)) + 1e-10)

        # 快速SSIM（简化版）
        if FAST_MODE:
            # 下采样计算SSIM（更快）
            step = 10
            ssim = structural_similarity(
                clutter_norm[::step, ::step],
                official_norm[::step, ::step],
                data_range=1.0
            )
        else:
            ssim = structural_similarity(clutter_norm, official_norm, data_range=1.0)

        # 相关系数
        valid = (clutter_norm.flatten() > 0.01) & (official_norm.flatten() > 0.01)
        if np.sum(valid) > 100:
            # 随机采样加速
            if FAST_MODE and np.sum(valid) > 10000:
                idx = np.random.choice(np.where(valid)[0], 10000, replace=False)
                corr = np.corrcoef(clutter_norm.flatten()[idx],
                                   official_norm.flatten()[idx])[0, 1]
            else:
                corr = np.corrcoef(clutter_norm.flatten()[valid],
                                   official_norm.flatten()[valid])[0, 1]
        else:
            corr = 0

        ssim_values.append(ssim)
        corr_values.append(corr)
        print(f"SSIM={ssim:.3f}")

    # 找到最优参数
    best_idx = np.argmax(ssim_values)
    best_param = param_range[best_idx]
    best_ssim = ssim_values[best_idx]

    # 计算Fisher信息（参数敏感性）
    dssim_dtheta = np.gradient(ssim_values, param_range)
    fisher_info = np.sum(dssim_dtheta ** 2)

    # 计算Cramer-Rao下界
    if fisher_info > 0:
        crlb = 1.0 / np.sqrt(fisher_info)
    else:
        crlb = np.inf

    print(f"\n  最优{param_name}: {best_param:.3f}")
    print(f"  最大SSIM: {best_ssim:.3f}")
    print(f"  Fisher信息: {fisher_info:.3e}")
    print(f"  Cramer-Rao下界: {crlb:.3e}")

    return {
        'param_range': param_range,
        'ssim_values': ssim_values,
        'corr_values': corr_values,
        'best_param': best_param,
        'best_ssim': best_ssim,
        'fisher_info': fisher_info,
        'crlb': crlb
    }


# ===========================
# 信道分解分析
# ===========================

def channel_decomposition_analysis(sim_results):
    """
    分析信道分量分解（论文Eq.1）
    H = H_LoS + H_Spec + H_Targ + H_Diff
    """
    H_spec = sim_results['H_spec']
    H_diff = sim_results['H_diff']

    # 功率贡献
    P_spec = np.mean(np.abs(H_spec) ** 2)
    P_diff = np.mean(np.abs(H_diff) ** 2)
    P_total = P_spec + P_diff

    # 相对贡献
    spec_contribution = P_spec / P_total * 100
    diff_contribution = P_diff / P_total * 100

    # 空间相关性
    spec_correlation_length = compute_correlation_length(np.abs(H_spec) ** 2)
    diff_correlation_length = compute_correlation_length(np.abs(H_diff) ** 2)

    print("\n=== 信道分解分析 ===")
    print(f"镜面反射贡献: {spec_contribution:.1f}%")
    print(f"漫散射贡献: {diff_contribution:.1f}%")
    print(f"镜面相关长度: {spec_correlation_length:.1f} 样本")
    print(f"漫散射相关长度: {diff_correlation_length:.1f} 样本")

    return {
        'P_spec': P_spec,
        'P_diff': P_diff,
        'spec_contribution': spec_contribution,
        'diff_contribution': diff_contribution
    }


def compute_correlation_length(signal_2d):
    """计算空间相关长度"""
    # 沿轨方向自相关
    center_row = signal_2d.shape[0] // 2
    signal_1d = signal_2d[center_row, :]

    if len(signal_1d) < 10:
        return 0

    # 归一化自相关
    correlation = np.correlate(signal_1d, signal_1d, mode='same')
    correlation = correlation / correlation[len(correlation) // 2]

    # 找到1/e点
    center = len(correlation) // 2
    for i in range(center, len(correlation)):
        if correlation[i] < 1 / np.e:
            return i - center

    return len(correlation) // 2


# ===========================
# 可视化函数
# ===========================

def create_theory_validation_plots(sim_results, identifiability_results, official_sim):
    """生成理论验证图表套件"""

    fig = plt.figure(figsize=(18, 12))

    # 1. 信道分解对比
    ax1 = plt.subplot(2, 3, 1)
    spec_power = np.abs(sim_results['H_spec']) ** 2
    spec_log = 10 * np.log10(spec_power + 1e-10)
    im1 = ax1.imshow(spec_log, aspect='auto', cmap='hot',
                     vmin=np.percentile(spec_log[spec_log > -100], 5),
                     vmax=np.percentile(spec_log[spec_log > -100], 95))
    ax1.set_title("H_Spec (Specular Component)")
    ax1.set_ylabel("Delay bin")
    plt.colorbar(im1, ax=ax1, label="dB")

    ax2 = plt.subplot(2, 3, 2)
    diff_power = np.abs(sim_results['H_diff']) ** 2
    diff_log = 10 * np.log10(diff_power + 1e-10)
    im2 = ax2.imshow(diff_log, aspect='auto', cmap='cool',
                     vmin=np.percentile(diff_log[diff_log > -100], 5),
                     vmax=np.percentile(diff_log[diff_log > -100], 95))
    ax2.set_title("H_Diff (Diffuse Component)")
    ax2.set_ylabel("Delay bin")
    plt.colorbar(im2, ax=ax2, label="dB")

    # 2. K因子映射
    ax3 = plt.subplot(2, 3, 3)
    K_map_dB = 10 * np.log10(sim_results['K_factor_map'] + 1e-10)
    ax3.plot(K_map_dB, 'b-', linewidth=1)
    ax3.axhline(y=sim_results['K_mean_dB'], color='r', linestyle='--',
                label=f"Mean: {sim_results['K_mean_dB']:.1f} dB")
    ax3.set_title("Rician K-factor (Eq.17)")
    ax3.set_xlabel("Along-track column")
    ax3.set_ylabel("K (dB)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 3. 参数可识别性
    ax4 = plt.subplot(2, 3, 4)
    param_name = list(identifiability_results.keys())[0]
    ident = identifiability_results[param_name]
    ax4.plot(ident['param_range'], ident['ssim_values'], 'o-', label='SSIM')
    ax4.axvline(x=ident['best_param'], color='r', linestyle='--',
                label=f"Best: {ident['best_param']:.3f}")
    ax4.set_title(f"Parameter Identifiability: {param_name}")
    ax4.set_xlabel(param_name)
    ax4.set_ylabel("SSIM")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 4. Fisher信息
    ax5 = plt.subplot(2, 3, 5)
    dssim = np.gradient(ident['ssim_values'], ident['param_range'])
    ax5.plot(ident['param_range'], dssim ** 2, 'g-', linewidth=2)
    ax5.fill_between(ident['param_range'], 0, dssim ** 2, alpha=0.3)
    ax5.set_title(f"Fisher Information")
    ax5.set_xlabel(param_name)
    ax5.set_ylabel("(∂SSIM/∂θ)²")
    ax5.grid(True, alpha=0.3)

    # 添加文本信息
    info_text = f"FIM: {ident['fisher_info']:.3e}\nCRLB: {ident['crlb']:.3e}"
    ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 5. 仿真vs官方对比
    ax6 = plt.subplot(2, 3, 6)
    total_power = np.abs(sim_results['H_spec'] + sim_results['H_diff']) ** 2

    # 计算相关性
    total_norm = total_power / (np.max(total_power) + 1e-10)
    official_norm = np.abs(official_sim) / (np.max(np.abs(official_sim)) + 1e-10)

    ssim = structural_similarity(total_norm, official_norm, data_range=1.0)

    # 散点密度图
    sample_points = min(5000, total_norm.size)
    idx = np.random.choice(total_norm.size, sample_points, replace=False)

    x = official_norm.flatten()[idx]
    y = total_norm.flatten()[idx]

    valid = (x > 0.01) & (y > 0.01)
    if np.sum(valid) > 100:
        xy = np.vstack([x[valid], y[valid]])
        z = gaussian_kde(xy)(xy)
        sc = ax6.scatter(x[valid], y[valid], c=z, s=1, cmap='viridis', alpha=0.5)
        plt.colorbar(sc, ax=ax6)

        # 添加1:1线
        ax6.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.5)

    ax6.set_title(f"Theory vs Official (SSIM={ssim:.3f})")
    ax6.set_xlabel("Official Clutter (Normalized)")
    ax6.set_ylabel("Theory Model (Normalized)")
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    ax6.grid(True, alpha=0.3)

    plt.suptitle("Theory Model Validation (Section II)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("theory_validation.png", dpi=150, bbox_inches='tight')
    print("\n✓ 理论验证图已保存: theory_validation.png")
    plt.show()


def create_environmental_mapping_table(identifiability_results):
    """
    创建环境参数映射表（对应论文Table I）
    """
    print("\n" + "=" * 70)
    print("Table I: Environmental Parameter Mapping Validation")
    print("=" * 70)

    headers = ["Parameter", "Best Value", "CRLB", "Fisher Info", "Max SSIM"]
    rows = []

    for param_name, results in identifiability_results.items():
        row = [
            param_name,
            f"{results['best_param']:.3f}",
            f"{results['crlb']:.3e}",
            f"{results['fisher_info']:.3e}",
            f"{results['best_ssim']:.3f}"
        ]
        rows.append(row)

    # 打印表格
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2
                  for i in range(len(headers))]

    # 打印表头
    header_line = "|".join(h.center(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # 打印数据行
    for row in rows:
        data_line = "|".join(str(v).center(w) for v, w in zip(row, col_widths))
        print(data_line)

    print("=" * 70)

    # 保存为CSV
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv("parameter_mapping_validation.csv", index=False)
    print("✓ 参数映射表已保存: parameter_mapping_validation.csv")


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
    """完整的理论模型验证流程"""

    print("=" * 70)
    print("火星ISAC信道模型验证")
    print("对应论文: Section II - The Unified Mars ISAC Channel Model")
    print("=" * 70)

    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    load_mola()

    if len(MOLA_TILES) == 0:
        print("错误: 无MOLA数据")
        return

    official_sim, geom_df, dt = load_sharad_data()
    n_rows, n_cols = official_sim.shape

    # 2. 设置环境参数
    print("\n[步骤2] 初始化环境参数向量 θ_env...")
    env_params = EnvironmentalParameters()
    print(f"  τ_vis = {env_params.tau_vis}")
    print(f"  ε_r = {env_params.epsilon_r}")
    print(f"  σ_h = {env_params.sigma_h} m")
    print(f"  l = {env_params.correlation_length} m")
    print(f"  S4 = {env_params.S4}")

    # 3. 理论模型前向仿真
    print("\n[步骤3] 执行理论模型前向仿真...")
    sim_results = forward_simulation_enhanced(geom_df, n_rows, n_cols, dt, env_params)

    # 4. 信道分解分析
    print("\n[步骤4] 信道分解分析...")
    decomp_results = channel_decomposition_analysis(sim_results)

    # 5. 参数可识别性分析
    print("\n[步骤5] 参数可识别性分析...")
    identifiability_results = {}

    # 分析关键参数
    for param in ['epsilon_r', 'sigma_h']:
        ident = parameter_identifiability_analysis(
            official_sim, geom_df, dt, param)
        identifiability_results[param] = ident

    # 6. 生成验证图表
    print("\n[步骤6] 生成理论验证图表...")
    create_theory_validation_plots(sim_results, identifiability_results, official_sim)

    # 7. 生成参数映射表
    print("\n[步骤7] 生成参数映射验证表...")
    create_environmental_mapping_table(identifiability_results)

    # 8. 计算最终指标
    print("\n" + "=" * 70)
    print("最终验证结果")
    print("=" * 70)

    # SSIM对比
    total_clutter = sim_results['clutter']
    clutter_norm = total_clutter / (np.max(total_clutter) + 1e-10)
    official_norm = np.abs(official_sim) / (np.max(np.abs(official_sim)) + 1e-10)

    final_ssim = structural_similarity(clutter_norm, official_norm, data_range=1.0)

    valid = (clutter_norm.flatten() > 0.01) & (official_norm.flatten() > 0.01)
    if np.sum(valid) > 100:
        final_corr = np.corrcoef(clutter_norm.flatten()[valid],
                                 official_norm.flatten()[valid])[0, 1]
    else:
        final_corr = 0

    print(f"理论模型 vs 官方杂波:")
    print(f"  SSIM: {final_ssim:.3f}")
    print(f"  相关系数: {final_corr:.3f}")
    print(f"  平均K因子: {sim_results['K_mean_dB']:.1f} dB")

    # 最优参数
    print(f"\n识别的最优参数:")
    for param_name, results in identifiability_results.items():
        print(f"  {param_name}: {results['best_param']:.3f} "
              f"(CRLB: {results['crlb']:.3e})")

    print("\n" + "=" * 70)
    print("✓ 理论模型验证完成！")
    print("=" * 70)

    print("\n生成的文件:")
    print("  - theory_validation.png (理论验证图)")
    print("  - parameter_mapping_validation.csv (参数映射表)")

    return {
        'sim_results': sim_results,
        'identifiability': identifiability_results,
        'final_ssim': final_ssim,
        'final_corr': final_corr
    }


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()