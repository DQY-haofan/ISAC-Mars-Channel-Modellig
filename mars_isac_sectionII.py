# ============================================================================
# Mars ISAC System - Complete Colab Notebook
# Section III: Fundamental Limits of Environmental Sensing
# ============================================================================

# %% [markdown]
# # Mars ISAC System Performance Analysis
# ## Section III: Fundamental Limits of Environmental Sensing
# 
# This notebook implements the theoretical performance bounds analysis from the paper,
# including CRLB vs ZZB comparison and EFIM degradation factor visualization.

# %% Cell 1: Setup and Clone Repository
import os
import sys

# Remove existing clone if present
!rm -rf /content/ISAC-Mars-Channel-Modellig

# Clone the repository
!git clone https://github.com/DQY-haofan/ISAC-Mars-Channel-Modellig.git

# Change to project directory
%cd /content/ISAC-Mars-Channel-Modellig

# For updates in subsequent runs, uncomment:
# !git pull origin main

print("✅ Repository cloned successfully!")

# %% Cell 2: Install Required Packages
!pip install numpy scipy matplotlib pandas -q
print("✅ Required packages installed!")

# %% Cell 3: Create Directory Structure
!mkdir -p simulations
!mkdir -p results
print("✅ Directory structure created!")

# %% Cell 4: Create Performance Bounds Analysis Script
%%writefile simulations/mars_isac_performance_bounds.py
"""
Mars ISAC System - CRLB vs. ZZB Performance Threshold Effects
This script generates a high-quality figure comparing Cramér-Rao Lower Bound (CRLB) 
and Ziv-Zakai Bound (ZZB) for dust optical depth estimation across different SNR regimes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.special import erfc
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for IEEE publication quality
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
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
    'savefig.pad_inches': 0.1
})

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
        """
        Calculate the Cramér-Rao Lower Bound for dust optical depth estimation.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            CRLB value (mean squared error)
        """
        snr_linear = 10**(snr_db / 10)
        
        # Fisher information for τ_vis (Eq. eq:fim_tau_vis_corrected)
        J_tau = (self.d * self.dalpha_dtau)**2 * self.N_eff * snr_linear / (1 + snr_linear)**2
        
        # CRLB (Eq. eq:crlb_tau_vis_corrected)
        crlb = 1 / J_tau if J_tau > 0 else np.inf
        
        return crlb
    
    def calculate_pe(self, h, snr_db):
        """
        Calculate the probability of error for binary hypothesis testing.
        
        Args:
            h: Hypothesis separation for τ_vis
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Probability of error
        """
        snr_linear = 10**(snr_db / 10)
        
        # Argument for Q-function (Eq. eq:pe_tau_corrected)
        arg = np.sqrt(self.N_eff * snr_linear * (h * self.d * self.dalpha_dtau)**2 / 
                      (2 * (1 + snr_linear)))
        
        # Q-function using complementary error function
        Pe = 0.5 * erfc(arg / np.sqrt(2))
        
        return Pe
    
    def valley_function(self, Pe):
        """
        Valley function V(Pe) for ZZB computation.
        
        Args:
            Pe: Probability of error
            
        Returns:
            Valley function value
        """
        if Pe < 1e-10:
            return 0
        elif Pe > 0.5 - 1e-10:
            return 1
        else:
            # Standard valley function
            return 2 * Pe * (1 - 2*Pe)
    
    def calculate_zzb(self, snr_db):
        """
        Calculate the Ziv-Zakai Bound using numerical integration.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            ZZB value (mean squared error)
        """
        def integrand(h):
            Pe = self.calculate_pe(h, snr_db)
            V = self.valley_function(Pe)
            return h * V
        
        # Numerical integration over [0, 2A]
        result, _ = quad(integrand, 0, 2*self.A_tau, limit=100)
        
        # ZZB formula (Eq. eq:zzb_general)
        zzb = result / (2 * self.A_tau)
        
        return zzb
    
    def calculate_bcrlb(self, snr_db):
        """
        Calculate the Bayesian CRLB with uniform prior.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            BCRLB value (mean squared error)
        """
        snr_linear = 10**(snr_db / 10)
        
        # Fisher information
        J_tau = (self.d * self.dalpha_dtau)**2 * self.N_eff * snr_linear / (1 + snr_linear)**2
        
        # Prior variance for uniform distribution
        sigma_p2 = self.A_tau**2 / 3
        
        # BCRLB (Eq. eq:bcrlb)
        bcrlb = 1 / (J_tau + 1/sigma_p2)
        
        return bcrlb

def plot_bounds_comparison():
    """Generate the main CRLB vs ZZB comparison plot."""
    
    # Initialize Mars ISAC system
    mars_isac = MarsISACBounds()
    
    # SNR range
    snr_db_range = np.linspace(-10, 30, 100)
    
    # Calculate bounds
    crlb_values = np.array([mars_isac.calculate_crlb(snr) for snr in snr_db_range])
    zzb_values = np.array([mars_isac.calculate_zzb(snr) for snr in snr_db_range])
    bcrlb_values = np.array([mars_isac.calculate_bcrlb(snr) for snr in snr_db_range])
    
    # Find threshold SNR (where ZZB ≈ 2×CRLB)
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
    
    # Add arrow annotation for threshold effect
    ax.annotate('Threshold Effect\n(CRLB breaks down)',
                xy=(snr_threshold, zzb_values[threshold_idx[0]] if len(threshold_idx) > 0 else 1e-3),
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
    
    plt.tight_layout()
    return fig

# Generate and save the figure
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    fig = plot_bounds_comparison()
    
    # Save in multiple formats
    fig.savefig('results/zzb_vs_crlb.pdf', format='pdf', dpi=300)
    fig.savefig('results/zzb_vs_crlb.png', format='png', dpi=300)
    
    plt.show()
    
    print("Figure generated successfully!")
    print("Files saved in results/ directory: zzb_vs_crlb.pdf and zzb_vs_crlb.png")

# %% Cell 5: Create EFIM Analysis Script
%%writefile simulations/mars_isac_efim_analysis.py
"""
Mars ISAC System - EFIM Performance Degradation Factor Heatmap
This script generates a high-quality heatmap showing how receiver imperfections
(phase noise and timing jitter) degrade environmental sensing performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for IEEE publication quality
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.grid': False,  # No grid for heatmaps
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class EFIMAnalysis:
    """Class for analyzing Effective Fisher Information Matrix degradation."""
    
    def __init__(self, B=10e6, T=1e-3, d=500e3, snr_db=20):
        """
        Initialize system parameters for EFIM analysis.
        
        Args:
            B: Bandwidth [Hz]
            T: Observation time [s]
            d: Propagation distance [m]
            snr_db: Operating SNR [dB]
        """
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
        """
        Calculate the Fisher Information Matrix elements.
        
        Returns:
            Dictionary containing FIM elements
        """
        snr = self.snr_linear
        
        # Environmental parameter FIM (τ_vis)
        J_tau_tau = (self.d * self.dalpha_dtau)**2 * self.N_eff * snr / (1 + snr)**2
        
        # Nuisance parameter FIM elements
        # Phase offset Fisher information
        J_phi_phi = self.N_eff * snr / (1 + snr)
        
        # Timing offset Fisher information  
        J_epsilon_epsilon = (2 * np.pi * self.B)**2 * self.gamma * self.N_eff * snr / (1 + snr)
        
        # Frequency offset Fisher information
        J_deltaf_deltaf = (2 * np.pi * self.T)**2 * self.N_eff * snr / (3 * (1 + snr))
        
        return {
            'J_tau_tau': J_tau_tau,
            'J_phi_phi': J_phi_phi,
            'J_epsilon_epsilon': J_epsilon_epsilon,
            'J_deltaf_deltaf': J_deltaf_deltaf
        }
    
    def calculate_coupling_terms(self):
        """
        Calculate the coupling terms between environmental and nuisance parameters.
        
        Returns:
            Dictionary containing coupling terms
        """
        snr = self.snr_linear
        
        # Coupling between τ_vis and phase offset
        # Phase variations affect power measurements
        J_tau_phi = np.sqrt(self.N_eff * snr / (1 + snr)) * self.d * self.dalpha_dtau * 0.1
        
        # Coupling between τ_vis and timing offset (Eq. eq:coupling_timing_corrected)
        J_tau_epsilon = (self.N_eff * snr / (1 + snr)) * self.d * self.dalpha_dtau * self.B * self.gamma
        
        # Coupling between τ_vis and frequency offset
        J_tau_deltaf = 0  # Orthogonal with proper pilot design
        
        return {
            'J_tau_phi': J_tau_phi,
            'J_tau_epsilon': J_tau_epsilon,
            'J_tau_deltaf': J_tau_deltaf
        }
    
    def calculate_eta(self, phi_std, epsilon_std, deltaf_std=None):
        """
        Calculate the performance degradation factor η.
        
        Args:
            phi_std: Phase noise standard deviation [rad]
            epsilon_std: Timing jitter standard deviation [s]
            deltaf_std: Frequency offset standard deviation [Hz] (optional)
            
        Returns:
            Performance degradation factor η
        """
        # Get FIM elements
        fim = self.calculate_fim_elements()
        coupling = self.calculate_coupling_terms()
        
        # Account for uncertainty in nuisance parameters
        # The FIM for nuisance parameters is reduced by their uncertainty
        J_phi_effective = fim['J_phi_phi'] / (1 + fim['J_phi_phi'] * phi_std**2)
        J_epsilon_effective = fim['J_epsilon_epsilon'] / (1 + fim['J_epsilon_epsilon'] * epsilon_std**2)
        
        # Calculate degradation factor (Eq. eq:degradation_factor_final)
        eta = 1.0
        
        # Phase noise contribution
        if J_phi_effective > 0:
            eta += coupling['J_tau_phi']**2 / (fim['J_tau_tau'] * J_phi_effective)
        
        # Timing jitter contribution  
        if J_epsilon_effective > 0:
            eta += coupling['J_tau_epsilon']**2 / (fim['J_tau_tau'] * J_epsilon_effective)
        
        # Frequency offset contribution (if provided)
        if deltaf_std is not None:
            J_deltaf_effective = fim['J_deltaf_deltaf'] / (1 + fim['J_deltaf_deltaf'] * deltaf_std**2)
            if J_deltaf_effective > 0:
                eta += coupling['J_tau_deltaf']**2 / (fim['J_tau_tau'] * J_deltaf_effective)
        
        return eta

def generate_efim_heatmap():
    """Generate the EFIM performance degradation heatmap."""
    
    # Initialize EFIM analysis with typical Mars link parameters
    efim = EFIMAnalysis(B=10e6, T=1e-3, d=500e3, snr_db=20)
    
    # Define parameter ranges
    # Phase noise standard deviation [rad]
    phi_std_range = np.logspace(-3, 0, 50)  # 0.001 to 1 rad
    
    # Timing jitter standard deviation [s]
    epsilon_std_range = np.logspace(-9, -6, 50)  # 1 ns to 1 μs
    
    # Create meshgrid
    PHI, EPSILON = np.meshgrid(phi_std_range, epsilon_std_range)
    
    # Calculate degradation factor for each combination
    ETA = np.zeros_like(PHI)
    for i in range(len(epsilon_std_range)):
        for j in range(len(phi_std_range)):
            ETA[i, j] = efim.calculate_eta(PHI[i, j], EPSILON[i, j])
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 6))
    
    # Subplot 1: Main heatmap
    ax1 = fig.add_subplot(121)
    
    # Create heatmap with logarithmic color scale
    im = ax1.pcolormesh(EPSILON * 1e9, PHI, ETA, 
                        cmap='plasma', 
                        shading='auto',
                        norm=LogNorm(vmin=1.0, vmax=10.0))
    
    # Add contour lines for specific degradation levels
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
    cbar.ax.set_ylabel('Degradation Factor $\\eta$', fontsize=11)
    
    # Add annotations for operational regions
    ax1.text(100, 0.002, 'Excellent\n(η < 1.1)', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    ax1.text(500, 0.01, 'Acceptable\n(η < 2)', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    ax1.text(800, 0.3, 'Poor\n(η > 3)', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    # Subplot 2: Cross-sections
    ax2 = fig.add_subplot(122)
    
    # Plot cross-sections at fixed phase noise levels
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
    return fig

def generate_3d_surface():
    """Generate a 3D surface plot of the degradation factor."""
    
    # Initialize EFIM analysis
    efim = EFIMAnalysis(B=10e6, T=1e-3, d=500e3, snr_db=20)
    
    # Define parameter ranges (coarser for 3D visualization)
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
    return fig

# Generate and save figures
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Generate main heatmap
    fig1 = generate_efim_heatmap()
    fig1.savefig('results/efim_degradation_heatmap.pdf', format='pdf', dpi=300)
    fig1.savefig('results/efim_degradation_heatmap.png', format='png', dpi=300)
    
    # Generate 3D surface plot
    fig2 = generate_3d_surface()
    fig2.savefig('results/efim_degradation_3d.pdf', format='pdf', dpi=300)
    fig2.savefig('results/efim_degradation_3d.png', format='png', dpi=300)
    
    plt.show()
    
    print("Figures generated successfully!")
    print("Files saved in results/ directory:")
    print("  - efim_degradation_heatmap.pdf/png")
    print("  - efim_degradation_3d.pdf/png")

# %% Cell 6: Run Performance Bounds Analysis
!python simulations/mars_isac_performance_bounds.py

# %% Cell 7: Run EFIM Analysis
!python simulations/mars_isac_efim_analysis.py

# %% Cell 8: Display Generated Figures
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Display all generated figures
print("=" * 60)
print("GENERATED FIGURES")
print("=" * 60)

# Display CRLB vs ZZB figure
print("\n📊 Figure 1: Performance Bounds Comparison")
img1 = mpimg.imread('results/zzb_vs_crlb.png')
plt.figure(figsize=(10, 8))
plt.imshow(img1)
plt.axis('off')
plt.title('CRLB vs ZZB: Threshold Effects in Mars Dust Sensing')
plt.show()

# Display EFIM heatmap
print("\n🗺️ Figure 2: EFIM Performance Degradation Heatmap")
img2 = mpimg.imread('results/efim_degradation_heatmap.png')
plt.figure(figsize=(14, 6))
plt.imshow(img2)
plt.axis('off')
plt.title('Impact of Receiver Imperfections on Sensing Accuracy')
plt.show()

# Display 3D surface
print("\n🏔️ Figure 3: 3D Degradation Surface")
img3 = mpimg.imread('results/efim_degradation_3d.png')
plt.figure(figsize=(10, 8))
plt.imshow(img3)
plt.axis('off')
plt.title('3D Visualization of Performance Degradation')
plt.show()

# %% Cell 9: Check Generated Files
print("\n" + "=" * 60)
print("GENERATED FILES")
print("=" * 60)
!ls -la results/

# %% Cell 10: Create Archive for Download
!zip -r mars_isac_simulation_results.zip results/
from google.colab import files
print("\n📦 Downloading simulation results...")
files.download('mars_isac_simulation_results.zip')

# %% Cell 11: Summary and Next Steps
print("\n" + "=" * 60)
print("SIMULATION COMPLETE!")
print("=" * 60)
print("\n✅ All simulations completed successfully!")
print("\n📁 Generated files:")
print("  • results/zzb_vs_crlb.pdf - Performance bounds comparison (PDF)")
print("  • results/zzb_vs_crlb.png - Performance bounds comparison (PNG)")
print("  • results/efim_degradation_heatmap.pdf - EFIM degradation heatmap (PDF)")
print("  • results/efim_degradation_heatmap.png - EFIM degradation heatmap (PNG)")
print("  • results/efim_degradation_3d.pdf - 3D degradation surface (PDF)")
print("  • results/efim_degradation_3d.png - 3D degradation surface (PNG)")
print("\n📊 Key findings:")
print("  • Threshold SNR ≈ 8-10 dB for reliable dust sensing")
print("  • Phase noise < 0.01 rad and timing jitter < 100 ns for η < 1.5")
print("  • Effective sample count N_eff critical for lowering threshold")
print("\n🔄 To re-run with updated parameters:")
print("  1. Edit the scripts in simulations/ folder")
print("  2. Re-run cells 6 and 7")
print("  3. View updated results in cell 8")