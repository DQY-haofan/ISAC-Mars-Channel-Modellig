#!/usr/bin/env python3
"""
Mars ISAC System - Section III: Fundamental Limits of Environmental Sensing
Improved visualization with larger fonts and cleaner layout
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.special import erfc
from matplotlib.colors import Normalize
import os
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication quality with LARGER FONTS
plt.rcParams.update({
    'font.size': 22,              # Increased from 11
    'font.family': 'sans-serif',
    'axes.labelsize': 22,         # Increased from 11
    'axes.titlesize': 22,         # Increased from 12
    'legend.fontsize': 18,        # Increased from 10
    'xtick.labelsize': 18,        # Increased from 10
    'ytick.labelsize': 18,        # Increased from 10
    'figure.figsize': (7, 4),
    'figure.dpi': 300,
    'lines.linewidth': 3,       # Slightly thicker
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
})

# ============================================================================
# Core Classes (unchanged)
# ============================================================================

class MarsISACBounds:
    """
    Computes theoretical performance bounds for Mars ISAC systems.
    """
    
    def __init__(self, B, T, d, H_dust=11e3, kappa=1.2, A_tau=0.5):
        """
        Initialize system parameters.
        
        Args:
            B: Bandwidth [Hz]
            T: Observation time [s]
            d: Propagation distance [m]
            H_dust: Dust scale height [m]
            kappa: Correlation penalty factor
            A_tau: Prior half-width for τ_vis
        """
        self.B = B
        self.T = T
        self.d = d
        self.H_dust = H_dust
        self.kappa = kappa
        self.A_tau = A_tau
        
        # Derived parameters
        self.N_eff = B * T / kappa
        self.dalpha_dtau = 1.0 / H_dust
        
    def estimate_threshold_snr(self):
        """
        Estimate the threshold SNR where ZZB starts to exceed CRLB.
        Returns SNR in dB.
        """
        C = (2.0 * self.A_tau * self.d * self.dalpha_dtau) * np.sqrt(self.N_eff / 2.0)
        if C <= 1.0:
            return -30.0  # Very low threshold
        s = 1.0 / C
        s = max(min(s, 0.99), 1e-6)  # Clamp to avoid numerical issues
        snr_linear = s / (1.0 - s)
        return 10.0 * np.log10(snr_linear)
    
    def calculate_crlb(self, snr_db):
        """Calculate CRLB with SNR² scaling."""
        snr_linear = 10**(snr_db / 10)
        J_tau = (self.d * self.dalpha_dtau)**2 * self.N_eff * (snr_linear**2) / (1 + snr_linear)**2
        return 1 / J_tau if J_tau > 0 else np.inf
    
    def calculate_pe(self, h, snr_db):
        """Calculate probability of error for binary hypothesis testing."""
        snr_linear = 10**(snr_db / 10)
        arg = np.sqrt(0.5 * self.N_eff) * (snr_linear / (1.0 + snr_linear)) * (h * self.d * self.dalpha_dtau)
        return 0.5 * erfc(arg / np.sqrt(2))
    
    def valley_filled_pe(self, h_array, snr_db):
        """Apply valley-filling to enforce monotonicity."""
        Pe_vals = np.array([self.calculate_pe(h, snr_db) for h in h_array])
        Pe_filled = np.copy(Pe_vals)
        for i in range(len(Pe_filled) - 2, -1, -1):
            Pe_filled[i] = max(Pe_filled[i], Pe_filled[i + 1])
        return Pe_filled
    
    def calculate_zzb(self, snr_db):
        """Calculate ZZB with valley-filling and triangular weighting."""
        h_vals = np.linspace(0, 2 * self.A_tau, 2000)
        Pe_filled = self.valley_filled_pe(h_vals, snr_db)
        weights = 1.0 - h_vals / (2 * self.A_tau)
        integrand = 0.5 * h_vals * weights * Pe_filled
        return simpson(integrand, h_vals)
    
    def calculate_bcrlb(self, snr_db):
        """Calculate Bayesian CRLB."""
        snr_linear = 10**(snr_db / 10)
        J_tau = (self.d * self.dalpha_dtau)**2 * self.N_eff * (snr_linear**2) / (1 + snr_linear)**2
        sigma_p2 = self.A_tau**2 / 3
        return 1 / (J_tau + 1/sigma_p2)

class EFIMAnalysis:
    """
    EFIM analysis for synchronization impairments.
    """
    
    def __init__(self, B, T, d, H_dust=11e3, snr_db=20, psi=0.02, gamma2=0.01):
        self.B = B
        self.T = T
        self.d = d
        self.H_dust = H_dust
        self.snr_linear = 10**(snr_db / 10)
        self.dalpha_dtau = 1.0 / H_dust
        self.kappa = 1.2
        self.N_eff = B * T / self.kappa
        self.psi = psi
        self.gamma2 = gamma2
    
    def calculate_eta(self, phi_std, epsilon_std):
        """Calculate degradation factor η."""
        snr = self.snr_linear
        
        # Environmental parameter FIM
        J_tau_tau = (self.d * self.dalpha_dtau)**2 * self.N_eff * (snr**2) / (1 + snr)**2
        
        # Nuisance parameters with pilot fraction
        J_phi_phi = self.psi * self.N_eff * snr / (1 + snr)
        J_epsilon_epsilon = self.psi * (2 * np.pi * self.B)**2 * self.gamma2 * self.N_eff * snr / (1 + snr)
        
        # Coupling terms
        J_tau_phi = 0.01 * self.d * self.dalpha_dtau * np.sqrt(self.N_eff * snr / (1 + snr))
        J_tau_epsilon = self.d * self.dalpha_dtau * self.B * np.sqrt(self.gamma2) * np.sqrt(self.N_eff * snr / (1 + snr))
        
        # Build matrices
        J_nui = np.array([[J_phi_phi, 0], [0, J_epsilon_epsilon]], dtype=float)
        
        # Add prior information
        if phi_std > 0:
            J_nui[0, 0] += 1.0 / (phi_std**2)
        if epsilon_std > 0:
            J_nui[1, 1] += 1.0 / (epsilon_std**2)
        
        J_tau_nui = np.array([[J_tau_phi, J_tau_epsilon]])
        J_nui_tau = J_tau_nui.T
        
        # Schur complement
        try:
            J_efim = J_tau_tau - (J_tau_nui @ np.linalg.inv(J_nui) @ J_nui_tau)[0, 0]
            eta = J_tau_tau / J_efim if J_efim > 0 else 1.0
        except:
            eta = 1.0
        
        return eta

# ============================================================================
# Plotting Functions with Improved Layout
# ============================================================================

def plot_bounds_comparison(params_dict, save_dir='results', subplot_label=''):
    """
    Generate CRLB vs ZZB comparison plot with cleaner layout.
    
    Args:
        params_dict: Dictionary with keys B, T, d, H_dust, A_tau, kappa
        save_dir: Directory to save figures
        subplot_label: Label like '(a)', '(b)', '(c)'
    """
    # Extract parameters
    B = params_dict.get('B', 10e6)
    T = params_dict.get('T', 1e-3)
    d = params_dict.get('d', 500e3)
    H_dust = params_dict.get('H_dust', 11e3)
    A_tau = params_dict.get('A_tau', 0.5)
    kappa = params_dict.get('kappa', 1.2)
    
    # Initialize system
    mars_isac = MarsISACBounds(B, T, d, H_dust, kappa, A_tau)
    
    # Estimate threshold and set SNR range
    snr_threshold_est = mars_isac.estimate_threshold_snr()
    snr_min = min(snr_threshold_est - 10, -30)
    snr_max = max(10, snr_threshold_est + 30)
    snr_db_range = np.linspace(snr_min, snr_max, 200)
    
    print(f"\n{subplot_label} Parameters:")
    print(f"  B={B/1e6:.2f} MHz, T={T*1e3:.2f} ms, d={d/1e3:.0f} km")
    print(f"  N_eff={mars_isac.N_eff:.1f}, Threshold SNR ≈ {snr_threshold_est:.1f} dB")
    
    # Calculate bounds
    crlb_values = np.array([mars_isac.calculate_crlb(snr) for snr in snr_db_range])
    zzb_values = np.array([mars_isac.calculate_zzb(snr) for snr in snr_db_range])
    bcrlb_values = np.array([mars_isac.calculate_bcrlb(snr) for snr in snr_db_range])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Plot bounds
    ax.semilogy(snr_db_range, crlb_values, 'b-', label='CRLB', linewidth=3)
    ax.semilogy(snr_db_range, zzb_values, 'r--', label='ZZB', linewidth=3)
    ax.semilogy(snr_db_range, bcrlb_values, 'g-.', label='BCRLB', linewidth=2.5, alpha=0.8)
    
    # Mark estimated threshold
    if snr_min <= snr_threshold_est <= snr_max:
        ax.axvline(x=snr_threshold_est, color='k', linestyle=':', alpha=0.3, linewidth=1.5)
    
    # Simplified title with subplot label
    title = f'{subplot_label} Mars Dust Sensing Performance Bounds'
    ax.set_title(title, fontsize=24, fontweight='bold', pad=15)
    
    # Labels
    ax.set_xlabel('SNR [dB]', fontsize=20)
    ax.set_ylabel('MSE for $\\tau_{vis}$', fontsize=20)
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    ax.legend(loc='best', fontsize=20, frameon=True, shadow=True)
    ax.set_xlim([snr_min, snr_max])
    ax.set_ylim([1e-8, 1e2])
    
    # Add parameter box (smaller, bottom right)
    info_text = f'B={B/1e6:.1f} MHz\nd={d/1e3:.0f} km\nThreshold≈{snr_threshold_est:.1f} dB'
    ax.text(0.97, 0.03, info_text, transform=ax.transAxes, fontsize=16,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = f'fig3_1_bounds_{subplot_label.strip("()")}'
    fig.savefig(os.path.join(save_dir, f'{filename}.pdf'), format='pdf', dpi=300)
    fig.savefig(os.path.join(save_dir, f'{filename}.png'), format='png', dpi=300)
    plt.show()
    
    return fig

def plot_efim_heatmap(params_dict, save_dir='results'):
    """Generate EFIM degradation heatmap with larger fonts."""
    B = params_dict.get('B', 10e6)
    T = params_dict.get('T', 1e-3)
    d = params_dict.get('d', 500e3)
    H_dust = params_dict.get('H_dust', 11e3)
    psi = params_dict.get('psi', 0.02)
    gamma2 = params_dict.get('gamma2', 0.01)
    
    efim = EFIMAnalysis(B, T, d, H_dust, snr_db=20, psi=psi, gamma2=gamma2)
    
    # Parameter ranges
    phi_std_range = np.logspace(-3, 0, 50)
    epsilon_std_range = np.logspace(-9, -6, 50)
    PHI, EPSILON = np.meshgrid(phi_std_range, epsilon_std_range)
    
    # Calculate degradation
    ETA = np.zeros_like(PHI)
    for i in range(len(epsilon_std_range)):
        for j in range(len(phi_std_range)):
            ETA[i, j] = efim.calculate_eta(PHI[i, j], EPSILON[i, j])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap
    im = ax1.pcolormesh(EPSILON * 1e9, PHI, ETA, cmap='plasma', 
                        shading='auto', norm=Normalize(vmin=1.0, vmax=2.0))
    
    # Contours
    levels = [1.05, 1.1, 1.2, 1.5, 2.0]
    CS = ax1.contour(EPSILON * 1e9, PHI, ETA, levels=levels, 
                     colors='white', linewidths=1.5, alpha=0.7)
    ax1.clabel(CS, inline=True, fontsize=11, fmt='%.2f')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Timing Jitter [ns]', fontsize=16)
    ax1.set_ylabel('Phase Noise [rad]', fontsize=16)
    ax1.set_title('EFIM Degradation Factor', fontsize=18, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('η', fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    
    # Cross-sections
    for phi_val, color, label in zip([0.001, 0.01, 0.1], 
                                     ['blue', 'green', 'red'],
                                     ['1 mrad', '10 mrad', '100 mrad']):
        eta_slice = [efim.calculate_eta(phi_val, eps) for eps in epsilon_std_range]
        ax2.semilogx(epsilon_std_range * 1e9, eta_slice, color=color, 
                    label=f'φ = {label}', linewidth=2.5)
    
    ax2.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, linewidth=1.5)
    ax2.set_xlabel('Timing Jitter [ns]', fontsize=16)
    ax2.set_ylabel('Degradation Factor η', fontsize=16)
    ax2.set_title('Cross-Sections', fontsize=18, fontweight='bold')
    ax2.grid(True, which='both', linestyle=':', alpha=0.3)
    ax2.legend(loc='best', fontsize=13)
    ax2.set_ylim([0.9, 2.5])
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'fig3_2_efim.pdf'), format='pdf', dpi=300)
    fig.savefig(os.path.join(save_dir, 'fig3_2_efim.png'), format='png', dpi=300)
    plt.show()
    
    return fig

# ============================================================================
# Parameter Configurations
# ============================================================================

def get_params_a():
    """Configuration (a) - Mission parameters."""
    return {
        'B': 10e6,        # 10 MHz
        'T': 1e-3,        # 1 ms
        'd': 500e3,       # 500 km
        'H_dust': 11e3,   # 11 km
        'A_tau': 0.5,     # Prior [0, 0.5]
        'kappa': 1.2
    }

def get_params_b():
    """Configuration (b) - Demonstration parameters."""
    return {
        'B': 100e3,       # 100 kHz
        'T': 0.5e-3,      # 0.5 ms
        'd': 50e3,        # 50 km
        'H_dust': 11e3,   # 11 km
        'A_tau': 0.5,     # Prior [0, 0.5]
        'kappa': 1.2
    }

def get_params_c():
    """Configuration (c) - Textbook parameters."""
    return {
        'B': 50e3,        # 50 kHz
        'T': 0.2e-3,      # 0.2 ms
        'd': 20e3,        # 20 km
        'H_dust': 11e3,   # 11 km
        'A_tau': 1.0,     # Prior [0, 1.0]
        'kappa': 1.2
    }

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Generate all figures with simplified labeling."""
    print("\n" + "=" * 70)
    print("MARS ISAC SYSTEM - PERFORMANCE BOUNDS ANALYSIS")
    print("=" * 70)
    
    save_dir = 'results'
    
    # Generate bounds comparison for different configurations
    configurations = [
        ('(a)', get_params_a()),
        ('(b)', get_params_b()),
        ('(c)', get_params_c())
    ]
    
    for label, params in configurations:
        plot_bounds_comparison(params, save_dir, label)
    
    # Generate EFIM heatmap
    print("\nGenerating EFIM analysis...")
    efim_params = get_params_a()  # Use configuration (a) for EFIM
    efim_params.update({'psi': 0.02, 'gamma2': 0.01})
    plot_efim_heatmap(efim_params, save_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • fig3_1_bounds_a.pdf/png")
    print("  • fig3_1_bounds_b.pdf/png")
    print("  • fig3_1_bounds_c.pdf/png")
    print("  • fig3_2_efim.pdf/png")

if __name__ == "__main__":
    main()