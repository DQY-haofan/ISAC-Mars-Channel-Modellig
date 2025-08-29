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
    # Generate main heatmap
    fig1 = generate_efim_heatmap()
    fig1.savefig('efim_degradation_heatmap.pdf', format='pdf', dpi=300)
    fig1.savefig('efim_degradation_heatmap.png', format='png', dpi=300)
    
    # Generate 3D surface plot
    fig2 = generate_3d_surface()
    fig2.savefig('efim_degradation_3d.pdf', format='pdf', dpi=300)
    fig2.savefig('efim_degradation_3d.png', format='png', dpi=300)
    
    plt.show()
    
    print("Figures generated successfully!")
    print("Files saved:")
    print("  - efim_degradation_heatmap.pdf/png")
    print("  - efim_degradation_3d.pdf/png")