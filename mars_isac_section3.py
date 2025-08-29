"""
Mars ISAC System - Section III: Fundamental Limits of Environmental Sensing
Combined analysis script for CRLB vs ZZB bounds and EFIM degradation factor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.special import erfc
from matplotlib.colors import LogNorm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for IEEE publication quality
# Handle font availability across different systems
try:
    # Try to use Times New Roman if available
    import matplotlib.font_manager as fm
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    if 'Times New Roman' in available_fonts:
        plt.rcParams['font.serif'] = ['Times New Roman']
    elif 'DejaVu Serif' in available_fonts:
        plt.rcParams['font.serif'] = ['DejaVu Serif']
    else:
        # Use default serif font
        plt.rcParams['font.serif'] = ['serif']
except:
    pass

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (7, 5),
    'figure.dpi': 300,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'text.usetex': False  # Disable LaTeX to avoid font issues
})

# Create results directory if it doesn't exist
def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created 'results' directory")
    return 'results'

# ============================================================================
# Part 1: CRLB vs ZZB Analysis
# ============================================================================

class MarsISACBounds:
    """Class for computing theoretical performance bounds for Mars ISAC systems."""
    
    def __init__(self, B=10e6, T=1e-3, d=500e3, H_dust=11e3, beta_ext=0.012):
        """
        Initialize Mars ISAC system parameters.
        
        Args:
            B: Bandwidth [Hz] (default: 10 MHz)
            T: Observation time [s] (default: 1 ms)
            d: Propagation distance [m] (default: 500 km)
            H_dust: Dust scale height [m] (default: 11 km)
            beta_ext: Mass extinction efficiency [m²/g] (default: 0.012)
        """
        self.B = B
        self.T = T
        self.d = d
        self.H_dust = H_dust
        self.beta_ext = beta_ext
        
        # Correlation penalty factor (κ ≥ 1)
        self.kappa = 1.2  # Accounting for filtering and multipath
        
        # Effective number of independent samples
        self.N_eff = B * T / self.kappa
        
        # Derivative of extinction coefficient w.r.t. τ_vis
        self.dalpha_dtau = beta_ext / H_dust  # ≈ 1.5e-4 m^-1
        
        # Prior range for τ_vis
        self.A_tau = 2.0  # Typical range [0, 2] for Mars dust optical depth
        
    def calculate_crlb(self, snr_db):
        """Calculate the Cramér-Rao Lower Bound for dust optical depth estimation."""
        snr_linear = 10**(snr_db / 10)
        
        # Fisher information for τ_vis
        J_tau = (self.d * self.dalpha_dtau)**2 * self.N_eff * snr_linear / (1 + snr_linear)**2
        
        # CRLB
        crlb = 1 / J_tau if J_tau > 0 else np.inf
        
        return crlb
    
    def calculate_pe(self, h, snr_db):
        """Calculate the probability of error for binary hypothesis testing."""
        snr_linear = 10**(snr_db / 10)
        
        # Argument for Q-function
        arg = np.sqrt(self.N_eff * snr_linear * (h * self.d * self.dalpha_dtau)**2 / 
                      (2 * (1 + snr_linear)))
        
        # Q-function using complementary error function
        Pe = 0.5 * erfc(arg / np.sqrt(2))
        
        return Pe
    
    def valley_function(self, Pe):
        """Valley function V(Pe) for ZZB computation."""
        if Pe < 1e-10:
            return 0
        elif Pe > 0.5 - 1e-10:
            return 1
        else:
            # Standard valley function
            return 2 * Pe * (1 - 2*Pe)
    
    def calculate_zzb(self, snr_db):
        """Calculate the Ziv-Zakai Bound using numerical integration."""
        def integrand(h):
            Pe = self.calculate_pe(h, snr_db)
            V = self.valley_function(Pe)
            return h * V
        
        # Numerical integration over [0, 2A]
        result, _ = quad(integrand, 0, 2*self.A_tau, limit=100)
        
        # ZZB formula
        zzb = result / (2 * self.A_tau)
        
        return zzb
    
    def calculate_bcrlb(self, snr_db):
        """Calculate the Bayesian CRLB with uniform prior."""
        snr_linear = 10**(snr_db / 10)
        
        # Fisher information
        J_tau = (self.d * self.dalpha_dtau)**2 * self.N_eff * snr_linear / (1 + snr_linear)**2
        
        # Prior variance for uniform distribution
        sigma_p2 = self.A_tau**2 / 3
        
        # BCRLB
        bcrlb = 1 / (J_tau + 1/sigma_p2)
        
        return bcrlb

# ============================================================================
# Part 2: EFIM Analysis
# ============================================================================

class EFIMAnalysis:
    """Class for analyzing Effective Fisher Information Matrix degradation."""
    
    def __init__(self, B=10e6, T=1e-3, d=500e3, snr_db=20):
        """Initialize system parameters for EFIM analysis."""
        self.B = B
        self.T = T
        self.d = d
        self.snr_linear = 10**(snr_db / 10)
        
        # Mars atmospheric parameters
        self.H_dust = 11e3  # Dust scale height [m]
        self.beta_ext = 0.012  # Mass extinction efficiency [m²/g]
        self.dalpha_dtau = self.beta_ext / self.H_dust
        
        # Correlation penalty factor
        self.kappa = 1.2
        self.N_eff = B * T / self.kappa
        
        # Spectrum shape factor for timing coupling
        self.gamma = 1/3  # For rectangular spectrum
        
    def calculate_fim_elements(self):
        """Calculate the Fisher Information Matrix elements."""
        snr = self.snr_linear
        
        # Environmental parameter FIM (τ_vis)
        J_tau_tau = (self.d * self.dalpha_dtau)**2 * self.N_eff * snr / (1 + snr)**2
        
        # Nuisance parameter FIM elements
        J_phi_phi = self.N_eff * snr / (1 + snr)
        J_epsilon_epsilon = (2 * np.pi * self.B)**2 * self.gamma * self.N_eff * snr / (1 + snr)
        J_deltaf_deltaf = (2 * np.pi * self.T)**2 * self.N_eff * snr / (3 * (1 + snr))
        
        return {
            'J_tau_tau': J_tau_tau,
            'J_phi_phi': J_phi_phi,
            'J_epsilon_epsilon': J_epsilon_epsilon,
            'J_deltaf_deltaf': J_deltaf_deltaf
        }
    
    def calculate_coupling_terms(self):
        """Calculate the coupling terms between environmental and nuisance parameters."""
        snr = self.snr_linear
        
        # Coupling terms (more realistic values)
        # Phase coupling: small but non-zero due to phase-induced power variations
        J_tau_phi = self.d * self.dalpha_dtau * np.sqrt(self.N_eff * snr / (1 + snr)) * 0.01
        
        # Timing coupling: depends on signal bandwidth and spectrum shape
        J_tau_epsilon = self.d * self.dalpha_dtau * self.B * np.sqrt(self.N_eff * snr / (1 + snr)) * self.gamma * 0.1
        
        # Frequency coupling: typically zero with proper pilot design
        J_tau_deltaf = 0  # Orthogonal with proper pilot design
        
        return {
            'J_tau_phi': J_tau_phi,
            'J_tau_epsilon': J_tau_epsilon,
            'J_tau_deltaf': J_tau_deltaf
        }
    
    def calculate_eta(self, phi_std, epsilon_std, deltaf_std=None):
        """Calculate the performance degradation factor η."""
        # Get FIM elements
        fim = self.calculate_fim_elements()
        coupling = self.calculate_coupling_terms()
        
        # Calculate degradation factor using the correct formula
        eta = 1.0
        
        # Phase noise contribution
        if fim['J_phi_phi'] > 0 and phi_std > 0:
            # The degradation depends on how much the phase uncertainty affects the measurement
            phase_factor = coupling['J_tau_phi']**2 / (fim['J_tau_tau'] * fim['J_phi_phi'])
            # Scale by the actual phase uncertainty
            eta += phase_factor / (1 + 1/(fim['J_phi_phi'] * phi_std**2))
        
        # Timing jitter contribution  
        if fim['J_epsilon_epsilon'] > 0 and epsilon_std > 0:
            # The degradation depends on timing uncertainty
            timing_factor = coupling['J_tau_epsilon']**2 / (fim['J_tau_tau'] * fim['J_epsilon_epsilon'])
            # Scale by the actual timing uncertainty
            eta += timing_factor / (1 + 1/(fim['J_epsilon_epsilon'] * epsilon_std**2))
        
        # Frequency offset contribution (if provided)
        if deltaf_std is not None and deltaf_std > 0:
            if fim['J_deltaf_deltaf'] > 0:
                freq_factor = coupling['J_tau_deltaf']**2 / (fim['J_tau_tau'] * fim['J_deltaf_deltaf'])
                eta += freq_factor / (1 + 1/(fim['J_deltaf_deltaf'] * deltaf_std**2))
        
        # Ensure eta doesn't exceed reasonable bounds
        return min(eta, 100.0)  # Cap at 100x degradation for numerical stability

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_bounds_comparison(save_dir):
    """Generate the main CRLB vs ZZB comparison plot."""
    print("Generating CRLB vs ZZB comparison plot...")
    
    # Initialize Mars ISAC system
    mars_isac = MarsISACBounds()
    
    # SNR range
    snr_db_range = np.linspace(-10, 30, 100)
    
    # Calculate bounds
    crlb_values = np.array([mars_isac.calculate_crlb(snr) for snr in snr_db_range])
    zzb_values = np.array([mars_isac.calculate_zzb(snr) for snr in snr_db_range])
    bcrlb_values = np.array([mars_isac.calculate_bcrlb(snr) for snr in snr_db_range])
    
    # Find threshold SNR
    ratio = zzb_values / (crlb_values + 1e-10)
    threshold_idx = np.where(ratio < 2)[0]
    if len(threshold_idx) > 0:
        snr_threshold = snr_db_range[threshold_idx[0]]
    else:
        snr_threshold = 30
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot bounds
    ax.semilogy(snr_db_range, crlb_values, 'b-', label='CRLB', linewidth=2)
    ax.semilogy(snr_db_range, zzb_values, 'r--', label='ZZB', linewidth=2)
    ax.semilogy(snr_db_range, bcrlb_values, 'g-.', label='BCRLB', linewidth=1.5, alpha=0.7)
    
    # Mark threshold SNR
    ax.axvline(x=snr_threshold, color='k', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add region annotations
    ax.text(-5, 1e-2, 'Prior\nRegion', fontsize=10, ha='center', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    ax.text(snr_threshold-5, 1e-4, 'Threshold\nRegion', fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.3))
    ax.text(25, 1e-6, 'Asymptotic\nRegion', fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    
    # Add arrow annotation
    if len(threshold_idx) > 0:
        ax.annotate('Threshold Effect\n(CRLB breaks down)',
                    xy=(snr_threshold, zzb_values[threshold_idx[0]]),
                    xytext=(snr_threshold-8, 1e-1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=9, color='red')
    
    # Labels and formatting
    ax.set_xlabel('Signal-to-Noise Ratio (SNR$_{rx}$) [dB]', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE) for $\\tau_{vis}$', fontsize=12)
    ax.set_title('Performance Bounds for Mars Dust Optical Depth Estimation', fontsize=13, fontweight='bold')
    
    # Grid and legend
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    
    # Set axis limits
    ax.set_xlim([-10, 30])
    ax.set_ylim([1e-7, 1e0])
    
    # Add system parameters text
    param_text = f'B = {mars_isac.B/1e6:.0f} MHz, T = {mars_isac.T*1e3:.0f} ms\n'
    param_text += f'd = {mars_isac.d/1e3:.0f} km, $N_{{eff}}$ = {mars_isac.N_eff:.0f}'
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save figure
    fig.savefig(os.path.join(save_dir, 'fig3_1_zzb_vs_crlb.pdf'), format='pdf', dpi=300)
    fig.savefig(os.path.join(save_dir, 'fig3_1_zzb_vs_crlb.png'), format='png', dpi=300)
    plt.close(fig)
    
    print(f"  Saved: {save_dir}/fig3_1_zzb_vs_crlb.pdf/png")
    return fig

def generate_efim_heatmap(save_dir):
    """Generate the EFIM performance degradation heatmap."""
    print("Generating EFIM degradation heatmap...")
    
    # Initialize EFIM analysis
    efim = EFIMAnalysis(B=10e6, T=1e-3, d=500e3, snr_db=20)
    
    # Define parameter ranges
    phi_std_range = np.logspace(-3, 0, 50)  # 0.001 to 1 rad
    epsilon_std_range = np.logspace(-9, -6, 50)  # 1 ns to 1 μs
    
    # Create meshgrid
    PHI, EPSILON = np.meshgrid(phi_std_range, epsilon_std_range)
    
    # Calculate degradation factor
    ETA = np.zeros_like(PHI)
    for i in range(len(epsilon_std_range)):
        for j in range(len(phi_std_range)):
            ETA[i, j] = efim.calculate_eta(PHI[i, j], EPSILON[i, j])
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 6))
    
    # Subplot 1: Main heatmap
    ax1 = fig.add_subplot(121)
    
    # Temporarily disable grid for heatmap
    original_grid = plt.rcParams['axes.grid']
    plt.rcParams['axes.grid'] = False
    
    # Create heatmap
    im = ax1.pcolormesh(EPSILON * 1e9, PHI, ETA, 
                        cmap='plasma', 
                        shading='auto',
                        norm=LogNorm(vmin=1.0, vmax=10.0))
    
    # Add contour lines
    contour_levels = [1.1, 1.5, 2.0, 3.0, 5.0]
    CS = ax1.contour(EPSILON * 1e9, PHI, ETA, 
                     levels=contour_levels, 
                     colors='white', 
                     linewidths=1.0,
                     linestyles='--',
                     alpha=0.7)
    ax1.clabel(CS, inline=True, fontsize=9, fmt='η=%.1f')
    
    # Set logarithmic scales
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Labels
    ax1.set_xlabel('Timing Jitter Std. Dev. [ns]', fontsize=12)
    ax1.set_ylabel('Phase Noise Std. Dev. [rad]', fontsize=12)
    ax1.set_title('EFIM Performance Degradation Factor ($\\eta$)', fontsize=13, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, label='Degradation Factor $\\eta$')
    
    # Add annotations
    ax1.text(100, 0.002, 'Excellent\n(η < 1.1)', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    ax1.text(500, 0.01, 'Acceptable\n(η < 2)', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    ax1.text(800, 0.3, 'Poor\n(η > 3)', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    # Restore grid setting
    plt.rcParams['axes.grid'] = original_grid
    
    # Subplot 2: Cross-sections
    ax2 = fig.add_subplot(122)
    
    # Plot cross-sections
    phase_levels = [0.001, 0.01, 0.1]
    colors = ['blue', 'green', 'red']
    
    for phi_val, color in zip(phase_levels, colors):
        eta_slice = []
        for eps_val in epsilon_std_range:
            eta_slice.append(efim.calculate_eta(phi_val, eps_val))
        ax2.semilogx(epsilon_std_range * 1e9, eta_slice, 
                    label=f'φ_std = {phi_val:.3f} rad',
                    color=color, linewidth=2)
    
    # Add reference lines
    ax2.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='No degradation')
    ax2.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='2× degradation')
    
    # Labels and formatting
    ax2.set_xlabel('Timing Jitter Std. Dev. [ns]', fontsize=12)
    ax2.set_ylabel('Degradation Factor $\\eta$', fontsize=12)
    ax2.set_title('Performance Degradation Cross-Sections', fontsize=13, fontweight='bold')
    ax2.grid(True, which='both', linestyle=':', alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_ylim([0.9, 10])
    
    # Add system parameters text
    param_text = f'System Parameters:\n'
    param_text += f'B = {efim.B/1e6:.0f} MHz\n'
    param_text += f'T = {efim.T*1e3:.0f} ms\n'
    param_text += f'd = {efim.d/1e3:.0f} km\n'
    param_text += f'SNR = 20 dB'
    ax2.text(0.65, 0.95, param_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(os.path.join(save_dir, 'fig3_2_efim_degradation.pdf'), format='pdf', dpi=300)
    fig.savefig(os.path.join(save_dir, 'fig3_2_efim_degradation.png'), format='png', dpi=300)
    plt.close(fig)
    
    print(f"  Saved: {save_dir}/fig3_2_efim_degradation.pdf/png")
    return fig

def generate_3d_surface(save_dir):
    """Generate a 3D surface plot of the degradation factor."""
    print("Generating 3D surface plot...")
    
    # Initialize EFIM analysis
    efim = EFIMAnalysis(B=10e6, T=1e-3, d=500e3, snr_db=20)
    
    # Define parameter ranges
    phi_std_range = np.logspace(-3, 0, 30)
    epsilon_std_range = np.logspace(-9, -6, 30)
    
    # Create meshgrid
    PHI, EPSILON = np.meshgrid(phi_std_range, epsilon_std_range)
    
    # Calculate degradation factor
    ETA = np.zeros_like(PHI)
    for i in range(len(epsilon_std_range)):
        for j in range(len(phi_std_range)):
            ETA[i, j] = efim.calculate_eta(PHI[i, j], EPSILON[i, j])
    
    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(np.log10(EPSILON * 1e9), np.log10(PHI), 
                           np.log10(ETA),
                           cmap='viridis',
                           alpha=0.9,
                           edgecolor='none')
    
    # Labels
    ax.set_xlabel('log₁₀(Timing Jitter [ns])', fontsize=11)
    ax.set_ylabel('log₁₀(Phase Noise [rad])', fontsize=11)
    ax.set_zlabel('log₁₀(Degradation Factor η)', fontsize=11)
    ax.set_title('3D View of EFIM Performance Degradation', fontsize=13, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('log₁₀(η)', fontsize=10)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(os.path.join(save_dir, 'fig3_3_efim_3d.pdf'), format='pdf', dpi=300)
    fig.savefig(os.path.join(save_dir, 'fig3_3_efim_3d.png'), format='png', dpi=300)
    plt.close(fig)
    
    print(f"  Saved: {save_dir}/fig3_3_efim_3d.pdf/png")
    return fig

def generate_summary_statistics(save_dir):
    """Generate and save summary statistics for Section III."""
    print("\nGenerating summary statistics...")
    
    # Initialize systems
    mars_isac = MarsISACBounds()
    efim = EFIMAnalysis()
    
    # Calculate key metrics
    stats = {
        'System Parameters': {
            'Bandwidth': f'{mars_isac.B/1e6:.1f} MHz',
            'Observation Time': f'{mars_isac.T*1e3:.1f} ms',
            'Link Distance': f'{mars_isac.d/1e3:.0f} km',
            'Dust Scale Height': f'{mars_isac.H_dust/1e3:.1f} km',
            'Extinction Coefficient': f'{mars_isac.beta_ext:.3f} m²/g',
            'Effective Samples': f'{mars_isac.N_eff:.0f}',
        },
        'Performance Bounds at SNR=20dB': {
            'CRLB(τ_vis)': f'{mars_isac.calculate_crlb(20):.2e}',
            'ZZB(τ_vis)': f'{mars_isac.calculate_zzb(20):.2e}',
            'BCRLB(τ_vis)': f'{mars_isac.calculate_bcrlb(20):.2e}',
        },
        'EFIM Degradation (typical)': {
            'Phase noise (0.01 rad)': f'{efim.calculate_eta(0.01, 0):.2f}',
            'Timing jitter (100 ns)': f'{efim.calculate_eta(0, 100e-9):.2f}',
            'Combined effect': f'{efim.calculate_eta(0.01, 100e-9):.2f}',
        }
    }
    
    # Save to text file
    stats_file = os.path.join(save_dir, 'section3_summary_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MARS ISAC SYSTEM - SECTION III SUMMARY STATISTICS\n")
        f.write("Fundamental Limits of Environmental Sensing\n")
        f.write("=" * 60 + "\n\n")
        
        for category, metrics in stats.items():
            f.write(f"{category}:\n")
            f.write("-" * 40 + "\n")
            for key, value in metrics.items():
                f.write(f"  {key:30s}: {value}\n")
            f.write("\n")
    
    print(f"  Saved: {save_dir}/section3_summary_statistics.txt")
    
    # Also print to console
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for category, metrics in stats.items():
        print(f"\n{category}:")
        for key, value in metrics.items():
            print(f"  {key:30s}: {value}")
    
    return stats

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("MARS ISAC SYSTEM - SECTION III ANALYSIS")
    print("Fundamental Limits of Environmental Sensing")
    print("=" * 60 + "\n")
    
    # Ensure results directory exists
    save_dir = ensure_results_dir()
    
    # Generate all figures
    print("\nGenerating figures...")
    print("-" * 40)
    
    # Figure 1: CRLB vs ZZB comparison
    fig1 = plot_bounds_comparison(save_dir)
    
    # Figure 2: EFIM degradation heatmap
    fig2 = generate_efim_heatmap(save_dir)
    
    # Figure 3: 3D surface plot
    fig3 = generate_3d_surface(save_dir)
    
    # Generate summary statistics
    stats = generate_summary_statistics(save_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"All results saved in: {os.path.abspath(save_dir)}/")
    print("=" * 60 + "\n")
    
    return save_dir

if __name__ == "__main__":
    # Run main analysis
    results_dir = main()
    
    # Display completion message
    print("Files generated:")
    print("  1. fig3_1_zzb_vs_crlb.pdf/png - Performance bounds comparison")
    print("  2. fig3_2_efim_degradation.pdf/png - EFIM degradation heatmap")
    print("  3. fig3_3_efim_3d.pdf/png - 3D surface visualization")
    print("  4. section3_summary_statistics.txt - Key metrics summary")
    print("\nAnalysis timestamp:", np.datetime64('now'))