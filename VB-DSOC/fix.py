# !/usr/bin/env python3
"""
独立完整修复方案 - 不依赖外部模块
直接生成可用的配置和标定结果
"""

import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

print("=" * 60)
print("V-B系统独立修复方案")
print("=" * 60)

# ========== 物理常数 ==========
AU = 1.495978707e11
c = 3e8
pi = np.pi


# ========== 完整的物理模型（独立版本） ==========

def antenna_gain_optical(D, lam, eta=1.0):
    """光学天线增益"""
    return eta * (pi * D / lam) ** 2


def free_space_loss_dB(R, lam):
    """自由空间损耗(dB)"""
    return 20 * np.log10(4 * pi * R / lam)


def pointing_efficiency_gaussian(jitter_urad, lam, Dtx):
    """指向效率（修正版）"""
    sigma_theta = jitter_urad * 1e-6  # μrad转rad
    theta_beam = lam / (pi * Dtx)
    return float(np.exp(-(sigma_theta / theta_beam) ** 2))


def cn2_hv57(h_m, v=21.0, A=1.7e-14):
    """HV5/7湍流廓线"""
    h = np.asarray(h_m)
    return (0.00594 * (v / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
            + 2.7e-16 * np.exp(-h / 1500.0) + A * np.exp(-h / 100.0))


def rytov_variance_downlink(lam, zenith_deg=30.0, v=21.0, A=1.7e-14, h_top=20000.0):
    """Rytov方差"""
    k = 2 * pi / lam
    zenith_rad = np.deg2rad(zenith_deg)
    z = np.linspace(0.0, h_top, 2001) / np.cos(zenith_rad)
    Cn2 = cn2_hv57(z * np.cos(zenith_rad), v=v, A=A)
    integrand = Cn2 * z ** (5 / 6)
    sigma_R2 = 1.23 * k ** (7 / 6) * np.trapz(integrand, z)
    return sigma_R2


def gg_params_from_rytov(sigmaR2):
    """Rytov方差转GG参数"""
    sigmaR2 = max(sigmaR2, 1e-10)
    exp_term_a = 0.49 * sigmaR2 / ((1 + 1.11 * sigmaR2 ** (12 / 5)) ** (7 / 6))
    exp_term_b = 0.51 * sigmaR2 / ((1 + 0.69 * sigmaR2 ** (12 / 5)) ** (5 / 6))
    a = np.exp(exp_term_a) - 1.0
    b = np.exp(exp_term_b) - 1.0
    alpha = 1.0 / max(a, 1e-6)
    beta = 1.0 / max(b, 1e-6)
    return alpha, beta


def calculate_snr_correct(au, params):
    """
    正确的SNR计算
    SNR = Pr/Pn (dB域: SNR_dB = Pr_dBW - Pn_dBW)
    """
    R = au * AU

    # 增益
    Gt = antenna_gain_optical(params['D_tx_m'], params['lambda_m'], params['eta_tx'])
    Gr = antenna_gain_optical(params['D_rx_m'], params['lambda_m'], params['eta_rx'])

    # 损耗
    Lfs = free_space_loss_dB(R, params['lambda_m'])
    eta_point = pointing_efficiency_gaussian(
        params['jitter_urad'], params['lambda_m'], params['D_tx_m']
    )

    # 接收功率 (dBW)
    Pt_dBW = 10 * np.log10(params['P_tx_W'])
    Gt_dB = 10 * np.log10(Gt)
    Gr_dB = 10 * np.log10(Gr)
    L_point_dB = -10 * np.log10(eta_point)

    Pr_dBW = Pt_dBW + Gt_dB + Gr_dB - Lfs - params['L_atm_dB'] - L_point_dB

    # SNR（关键：减去噪声功率）
    noise_dBW = params.get('noise_factor_dB', -200)
    SNR_dB = Pr_dBW - noise_dBW

    return SNR_dB, Pr_dBW, Lfs


def outage_probability_gg(au, snr_th_dB, params, Nmc=20000):
    """计算中断概率（独立版本）"""
    # 平均SNR
    snr_bar, _, _ = calculate_snr_correct(au, params)

    # 湍流参数
    sigmaR2 = rytov_variance_downlink(
        params['lambda_m'],
        params.get('zenith_deg', 30.0),
        params.get('hv57_v', 21.0),
        params.get('hv57_A', 1.7e-14)
    )
    alpha, beta = gg_params_from_rytov(sigmaR2)

    # Monte Carlo
    X = np.random.gamma(alpha, 1 / alpha, Nmc)
    Y = np.random.gamma(beta, 1 / beta, Nmc)
    I = X * Y

    # 瞬时SNR
    snr_instant_dB = snr_bar + 10 * np.log10(I)

    # 中断概率
    p_out = float(np.mean(snr_instant_dB < snr_th_dB))

    return p_out, snr_bar, (alpha, beta)


# ========== 主程序 ==========

# 步骤1：设置参数
print("\n[1] 系统参数配置")

params = {
    'lambda_m': 1.55e-6,
    'D_tx_m': 0.22,
    'D_rx_m': 5.1,
    'eta_tx': 0.45,
    'eta_rx': 0.60,
    'P_tx_W': 4.0,
    'jitter_urad': 1.0,
    'L_atm_dB': 2.0,
    'zenith_deg': 30.0,
    'hv57_v': 21.0,
    'hv57_A': 1.7e-14,
    'noise_factor_dB': -200,  # 关键参数！
    'mc_samples': 50000
}

print(f"  发射功率: {params['P_tx_W']} W")
print(f"  口径: {params['D_tx_m']}m(Tx) / {params['D_rx_m']}m(Rx)")
print(f"  波长: {params['lambda_m'] * 1e9:.0f} nm")
print(f"  噪声基底: {params['noise_factor_dB']} dBW")

# 步骤2：验证SNR计算
print("\n[2] SNR计算验证")
print("\n距离(AU) | Pr(dBW) | SNR(dB) | Lfs(dB)")
print("-" * 50)

test_aus = [0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
for au in test_aus:
    snr, pr, lfs = calculate_snr_correct(au, params)
    print(f"{au:7.1f}  | {pr:7.1f} | {snr:7.1f} | {lfs:6.1f}")

# 步骤3：标定SNR门限
print("\n[3] SNR门限标定")


def calibrate_threshold(target_au, target_pout, params):
    """标定SNR门限"""
    print(f"  目标: P_out={target_pout:.1e} @ {target_au} AU")

    def objective(snr_th):
        pout, _, _ = outage_probability_gg(target_au, snr_th, params, Nmc=10000)
        return abs(np.log10(max(pout, 1e-10)) - np.log10(target_pout))

    # 先估算合理范围
    snr_at_target, _, _ = calculate_snr_correct(target_au, params)

    # 搜索
    result = minimize_scalar(
        objective,
        bounds=(snr_at_target - 20, snr_at_target + 20),
        method='bounded'
    )

    # 验证
    pout_check, snr_bar, (alpha, beta) = outage_probability_gg(
        result.x, result.x, params, Nmc=20000
    )

    print(f"  标定结果: SNR_th={result.x:.2f} dB")
    print(f"  平均SNR: {snr_bar:.2f} dB")
    print(f"  湍流参数: α={alpha:.1f}, β={beta:.1f}")

    return result.x


# 标定两个速率
print("\n25 Mbps @ 1.5 AU:")
snr_th_25 = calibrate_threshold(1.5, 0.001, params)

print("\n267 Mbps @ 0.3 AU:")
snr_th_267 = calibrate_threshold(0.3, 0.001, params)

# 步骤4：测试P_out曲线
print("\n[4] P_out曲线测试")

au_test = np.linspace(0.2, 2.5, 15)
pouts_25 = []
pouts_267 = []

print("\n计算中断概率...")
for au in au_test:
    pout_25, _, _ = outage_probability_gg(au, snr_th_25, params, Nmc=20000)
    pout_267, _, _ = outage_probability_gg(au, snr_th_267, params, Nmc=20000)
    pouts_25.append(pout_25)
    pouts_267.append(pout_267)

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：P_out曲线
ax1.semilogy(au_test, np.maximum(pouts_25, 1e-6), 'b-', linewidth=2, label='25 Mbps')
ax1.semilogy(au_test, np.maximum(pouts_267, 1e-6), 'r--', linewidth=2, label='267 Mbps')
ax1.plot(1.5, 0.001, 'bo', markersize=10, label='25Mbps锚点')
ax1.plot(0.3, 0.001, 'ro', markersize=10, label='267Mbps锚点')
ax1.set_xlabel('距离 (AU)')
ax1.set_ylabel('中断概率 P_out')
ax1.set_title('修复后的P_out曲线')
ax1.grid(True, which='both', alpha=0.3)
ax1.legend()
ax1.set_ylim([1e-6, 1])

# 右图：SNR vs 距离
ax2.plot(test_aus, [calculate_snr_correct(au, params)[0] for au in test_aus],
         'g-', linewidth=2, label='平均SNR')
ax2.axhline(y=snr_th_25, color='b', linestyle='--', label=f'25Mbps门限={snr_th_25:.1f}dB')
ax2.axhline(y=snr_th_267, color='r', linestyle='--', label=f'267Mbps门限={snr_th_267:.1f}dB')
ax2.set_xlabel('距离 (AU)')
ax2.set_ylabel('SNR (dB)')
ax2.set_title('SNR vs 距离')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle('V-B系统修复验证', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('standalone_fix_results.png', dpi=150)
print("\n图表已保存: standalone_fix_results.png")
plt.show()

# 步骤5：保存配置
print("\n[5] 保存修复配置")

# 基础配置
base_config = params.copy()
base_config['au_grid'] = {'start': 0.2, 'stop': 3.0, 'num': 80}
base_config['anchors'] = [
    {'name': 'Near Demo', 'au': 0.3, 'rate_bps': 267000000, 'pout_target': 0.001},
    {'name': 'Far Demo', 'au': 1.5, 'rate_bps': 25000000, 'pout_target': 0.001}
]
base_config['output'] = {
    'save_figures': True,
    'figure_dpi': 200,
    'save_data': True,
    'verbose': True
}

with open('dsoc_params_standalone.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(base_config, f, default_flow_style=False)
print("✓ 基础配置已保存: dsoc_params_standalone.yaml")

# 标定结果
calibrated_config = base_config.copy()
calibrated_config['calibrated_snr_thresholds'] = {
    '25Mbps': float(snr_th_25),
    '267Mbps': float(snr_th_267)
}

with open('vb_calibrated_standalone.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(calibrated_config, f, default_flow_style=False)
print("✓ 标定结果已保存: vb_calibrated_standalone.yaml")

# 总结
print("\n" + "=" * 60)
print("修复完成！")
print("=" * 60)
print("\n关键修复:")
print(f"✓ SNR正确计算: Pr - Pn (噪声={params['noise_factor_dB']}dBW)")
print(f"✓ 25Mbps门限: {snr_th_25:.1f} dB")
print(f"✓ 267Mbps门限: {snr_th_267:.1f} dB")
print(f"✓ P_out动态范围: {min(pouts_25):.2e} ~ {max(pouts_25):.2e}")

print("\n验证成功指标:")
if 70 < snr_th_25 < 100 and 90 < snr_th_267 < 110:
    print("✓ SNR门限在合理范围")
else:
    print("⚠️ SNR门限可能需要调整")

if min(pouts_25) < 0.01 and max(pouts_25) > 0.1:
    print("✓ P_out有良好动态范围")
else:
    print("⚠️ P_out动态范围有限")

print("\n后续步骤:")
print("1. 查看生成的图表: standalone_fix_results.png")
print("2. 使用生成的配置文件:")
print("   cp dsoc_params_standalone.yaml dsoc_params.yaml")
print("   cp vb_calibrated_standalone.yaml vb_calibrated_params.yaml")
print("3. 重新运行主程序（如果主程序已更新支持noise_factor_dB）")
print("=" * 60)