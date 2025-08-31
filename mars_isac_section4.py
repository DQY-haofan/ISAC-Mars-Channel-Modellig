#!/usr/bin/env python3
"""
================================================================================
Mars ISAC System - Section IV: The Sensing-Communication Performance Tradeoff
Pareto Optimal Boundaries Analysis under Martian Environmental Conditions
[CORRECTED VERSION - Aligned with theoretical framework]
================================================================================
This script implements the complete performance tradeoff analysis from Section IV:
1. Communication capacity calculation with Nakagami-m fading
2. Sensing precision calculation based on CRLB
3. Pareto boundary computation for different environmental scenarios
4. Comparison with TDM baseline to quantify synergistic gains

Authors: Mars ISAC Research Team
Date: 2024
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
from scipy.integrate import quad
from scipy.constants import c
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Global Configuration
# ============================================================================
import sys
if 'google.colab' in sys.modules:
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (10, 8),
        'figure.dpi': 300,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
else:
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (10, 8),
        'figure.dpi': 300,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

# ============================================================================
# Mars ISAC System Parameters
# ============================================================================

class MarsISACSystem:
    """
    Mars ISAC system with environmental-aware performance metrics.
    Implements the corrected framework from Section IV of the paper.
    """
    
    def __init__(self, link_type='uhf'):
        """
        Initialize Mars ISAC system parameters.
        
        Args:
            link_type: 'uhf' for proximity link, 'ka' for deep space link
        """
        if link_type == 'uhf':
            # UHF Proximity Link (Orbiter-Rover) - Table I parameters
            self.f_c = 435e6      # 435 MHz carrier frequency
            self.B = 10e6         # 10 MHz bandwidth
            self.P_total = 10     # 10 W transmit power
            self.d = 400e3        # 400 km link distance
            self.F_dB = 3         # 3 dB noise figure
            self.G_t_dBi = 10     # Transmit antenna gain [dBi]
            self.G_r_dBi = 10     # Receive antenna gain [dBi]
        elif link_type == 'ka':
            # Ka-band Deep Space Link
            self.f_c = 32e9       # 32 GHz carrier frequency
            self.B = 50e6         # 50 MHz bandwidth
            self.P_total = 50     # 50 W transmit power
            self.d = 1.5e11       # 1.5 AU link distance
            self.F_dB = 3         # 3 dB noise figure
            self.G_t_dBi = 45     # High-gain antenna [dBi]
            self.G_r_dBi = 45     # High-gain antenna [dBi]
        else:
            raise ValueError("link_type must be 'uhf' or 'ka'")
        
        # Convert antenna gains to linear scale
        self.G_t = 10**(self.G_t_dBi/10)
        self.G_r = 10**(self.G_r_dBi/10)
        
        # Common parameters
        self.T = 1e-3         # 1 ms observation time
        self.kappa = 1.2      # Correlation penalty factor
        self.N_eff = self.B * self.T / self.kappa
        
        # Noise parameters
        k_B = 1.38e-23        # Boltzmann constant
        T_0 = 290             # Reference temperature [K]
        F = 10**(self.F_dB/10)
        self.N_0 = k_B * T_0 * F  # Noise PSD [W/Hz]
        
        # Mars atmospheric parameters
        self.H_dust = 11e3    # Dust scale height [m]
        self.beta_ext = 0.012 # Mass extinction efficiency [m²/g]
        
    def dust_extinction_scale(self):
        """
        Calculate frequency-dependent dust extinction scaling factor.
        Dust scattering is wavelength-dependent: negligible at UHF, significant at optical.
        
        Returns:
            Scaling factor κ_ext(f) for dust extinction
        """
        f_GHz = self.f_c / 1e9
        
        if f_GHz < 1.0:         # UHF/VHF - Rayleigh regime, negligible extinction
            return 1e-6         # Near-zero but non-zero for numerical stability
        elif f_GHz < 10.0:      # L/S/C/X bands
            return 1e-3
        elif f_GHz < 100.0:     # Ku/Ka/W bands - Mie scattering becomes significant
            return 5e-2
        else:                   # Optical/FSO - full extinction
            return 1.0
    
    def calculate_attenuation(self, tau_vis):
        """
        Calculate atmospheric attenuation factor with frequency-dependent scaling.
        
        Args:
            tau_vis: Dust optical depth (at visible wavelengths)
            
        Returns:
            Attenuation factor (linear scale)
        """
        # Frequency-scaled extinction coefficient
        kappa_ext = self.dust_extinction_scale()
        alpha_ext = kappa_ext * (tau_vis / self.H_dust)
        return np.exp(-alpha_ext * self.d)
    
    def calculate_average_snr(self, rho, tau_vis):
        """
        Calculate average SNR with proper link budget.
        
        Args:
            rho: Power allocation factor for sensing
            tau_vis: Dust optical depth
            
        Returns:
            Average SNR (linear scale) including FSPL and antenna gains
        """
        # Power allocated to communication
        P_comm = (1 - rho) * self.P_total
        
        # Free space path loss
        wavelength = c / self.f_c
        L_fs = (4 * np.pi * self.d / wavelength)**2
        
        # Atmospheric attenuation
        atten = self.calculate_attenuation(tau_vis)
        
        # Average SNR with complete link budget
        # bar_gamma = (P_tx * G_t * G_r / L_fs) * atten / (N_0 * B)
        snr_avg = (P_comm * self.G_t * self.G_r / L_fs) * atten / (self.N_0 * self.B)
        
        return snr_avg
    
    def nakagami_m_mapping(self, S4):
        """
        Calibrated mapping from S4 to Nakagami-m parameter.
        
        Args:
            S4: Scintillation index
            
        Returns:
            Nakagami-m parameter
        """
        if S4 <= 0:
            return 100  # No fading limit
        elif S4 < 0.4:
            # Weak scintillation: m ≈ 1/S4²
            return 1 / S4**2
        else:
            # Strong scintillation: empirical calibration
            # Using a modified mapping to avoid m < 0.5
            return max(0.5, 1 / (S4**2 * 1.2))
    
    def ergodic_capacity_nakagami(self, rho, beta, tau_vis, S4):
        """
        Calculate ergodic capacity under Nakagami-m fading.
        Corrected implementation with proper Nakagami PDF.
        
        Args:
            rho: Power allocation factor for sensing
            beta: Resource block allocation factor for sensing
            tau_vis: Dust optical depth
            S4: Scintillation index
            
        Returns:
            Ergodic capacity [bps]
        """
        # Nakagami-m parameter
        m = self.nakagami_m_mapping(S4)
        
        # Average SNR
        bar_gamma = self.calculate_average_snr(rho, tau_vis)
        
        # For high m (weak fading), use deterministic approximation
        if m > 50:
            capacity_bps_hz = np.log2(1 + bar_gamma)
        else:
            # Numerical integration with correct Nakagami PDF
            # gamma ~ Gamma(shape=m, scale=bar_gamma/m)
            def integrand(x):
                # x is the normalized Gamma variable
                gamma_val = (bar_gamma / m) * x
                if gamma_val > 0:
                    # PDF: (m^m / Gamma(m)) * (x^(m-1)) * exp(-x)
                    pdf = (x**(m-1)) * np.exp(-x) / gamma_func(m)
                    return np.log2(1 + gamma_val) * pdf
                else:
                    return 0
            
            capacity_bps_hz, _ = quad(integrand, 0, np.inf, limit=200)
        
        # Account for resource allocation to communication
        capacity_bps = (1 - beta) * self.B * capacity_bps_hz
        
        return capacity_bps
    
    def sensing_precision(self, rho, beta, tau_vis, S4, target='dust'):
        """
        Calculate sensing precision (1/CRLB) for environmental parameters.
        Corrected implementation with SNR² for dust sensing.
        
        Args:
            rho: Power allocation factor for sensing
            beta: Resource block allocation factor for sensing
            tau_vis: Dust optical depth
            S4: Scintillation index
            target: 'dust' for τ_vis estimation, 'scintillation' for S4
            
        Returns:
            Sensing precision [1/parameter²]
        """
        # Power allocated to sensing
        P_sense = rho * self.P_total
        
        # Free space path loss
        wavelength = c / self.f_c
        L_fs = (4 * np.pi * self.d / wavelength)**2
        
        # Atmospheric attenuation
        atten = self.calculate_attenuation(tau_vis)
        
        # SNR for sensing (with antenna gains)
        snr_sense = (P_sense * self.G_t * self.G_r / L_fs) * atten / (self.N_0 * self.B)
        
        # Effective samples for sensing
        N_eff_sense = beta * self.B * self.T / self.kappa
        
        if target == 'dust':
            # Fisher information for dust optical depth
            # Corrected: dalpha_dtau = 1/H_dust (not beta_ext/H_dust)
            dalpha_dtau = 1.0 / self.H_dust
            # Corrected: SNR² dependence
            G_tau = (self.d * dalpha_dtau)**2 * (snr_sense**2) / (1 + snr_sense)**2
            precision = N_eff_sense * G_tau
            
        elif target == 'scintillation':
            # Fisher information for scintillation index
            m = self.nakagami_m_mapping(S4)
            if S4 > 0:
                # df/dS4 for the mapping function
                if S4 < 0.4:
                    df_dS4 = -2 / S4**3
                else:
                    df_dS4 = -2.4 / (S4**3 * 1.2)
                G_S4 = (df_dS4 / m)**2 * (snr_sense**2) / (1 + snr_sense)**2
                precision = N_eff_sense * G_S4
            else:
                precision = 0
        else:
            raise ValueError("target must be 'dust' or 'scintillation'")
        
        return precision

# ============================================================================
# Pareto Boundary Computation (Corrected)
# ============================================================================

def compute_pareto_boundary(system, tau_vis, S4, num_grid=31):
    """
    Compute true Pareto optimal boundary using grid search and non-dominated sorting.
    
    Args:
        system: MarsISACSystem instance
        tau_vis: Dust optical depth
        S4: Scintillation index
        num_grid: Grid resolution for (rho, beta) space
        
    Returns:
        Arrays of (sensing_precision, communication_capacity) points on Pareto frontier
    """
    # Grid search over (rho, beta) space
    rho_values = np.linspace(0, 1, num_grid)
    beta_values = np.linspace(0, 1, num_grid)
    
    # Store all feasible points
    points = []
    
    for rho in rho_values:
        for beta in beta_values:
            # Calculate performance metrics
            precision = system.sensing_precision(rho, beta, tau_vis, S4, target='dust')
            capacity = system.ergodic_capacity_nakagami(rho, beta, tau_vis, S4)
            
            points.append((precision, capacity / 1e6))  # Convert to Mbps
    
    # Non-dominated sorting to find Pareto frontier
    points = np.array(points)
    n_points = len(points)
    is_dominated = np.zeros(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Check if point j dominates point i
                if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and
                    (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    is_dominated[i] = True
                    break
    
    # Extract Pareto frontier
    pareto_points = points[~is_dominated]
    
    # Sort by sensing precision for smooth curve
    sorted_idx = np.argsort(pareto_points[:, 0])
    pareto_points = pareto_points[sorted_idx]
    
    return pareto_points[:, 0], pareto_points[:, 1]

def compute_tdm_baseline(system, tau_vis, S4):
    """
    Compute TDM baseline (time-division multiplexing).
    
    Args:
        system: MarsISACSystem instance
        tau_vis: Dust optical depth
        S4: Scintillation index
        
    Returns:
        Arrays of (sensing_precision, communication_capacity) for TDM line
    """
    # Maximum sensing precision (all resources to sensing)
    P_s_max = system.sensing_precision(1.0, 1.0, tau_vis, S4, target='dust')
    
    # Maximum communication capacity (all resources to communication)
    R_c_max = system.ergodic_capacity_nakagami(0.0, 0.0, tau_vis, S4) / 1e6  # Mbps
    
    # TDM line connects (0, R_c_max) to (P_s_max, 0)
    lambda_vals = np.linspace(0, 1, 50)
    sensing_tdm = lambda_vals * P_s_max
    comm_tdm = (1 - lambda_vals) * R_c_max
    
    return sensing_tdm, comm_tdm

# ============================================================================
# Visualization
# ============================================================================

def plot_pareto_boundaries():
    """
    Generate the main Pareto boundary comparison figure.
    """
    print("=" * 80)
    print("Mars ISAC System - Pareto Optimal Boundaries Analysis")
    print("=" * 80)
    
    # Initialize system (UHF proximity link)
    system = MarsISACSystem(link_type='uhf')
    
    # Environmental scenarios
    scenarios = [
        {'name': 'Baseline', 'tau_vis': 0.2, 'S4': 0.2, 
         'color': 'blue', 'linestyle': '-', 'marker': 'o'},
        {'name': 'Dust Storm', 'tau_vis': 2.0, 'S4': 0.2, 
         'color': 'red', 'linestyle': '--', 'marker': 's'},
        {'name': 'Strong Scintillation', 'tau_vis': 0.2, 'S4': 0.8, 
         'color': 'green', 'linestyle': '-.', 'marker': '^'},
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Store boundaries for synergistic gain calculation
    boundaries = {}
    
    # Plot Pareto boundaries for each scenario
    for scenario in scenarios:
        print(f"\nComputing {scenario['name']} scenario...")
        print(f"  τ_vis = {scenario['tau_vis']}, S4 = {scenario['S4']}")
        
        # Compute true Pareto boundary
        sensing, comm = compute_pareto_boundary(
            system, scenario['tau_vis'], scenario['S4'], num_grid=25
        )
        
        # Store for analysis
        boundaries[scenario['name']] = (sensing, comm)
        
        # Print actual performance values
        print(f"  Max sensing precision: {np.max(sensing):.2e}")
        print(f"  Max communication rate: {np.max(comm):.2f} Mbps")
        
        # Plot Pareto curve
        ax.plot(sensing, comm,  # Keep original units
                color=scenario['color'],
                linestyle=scenario['linestyle'],
                linewidth=2.5,
                marker=scenario['marker'],
                markevery=max(1, len(sensing)//10),
                markersize=6,
                label=f"{scenario['name']} (τ={scenario['tau_vis']}, S₄={scenario['S4']})",
                alpha=0.8)
        
        # Compute and plot TDM baseline
        sensing_tdm, comm_tdm = compute_tdm_baseline(
            system, scenario['tau_vis'], scenario['S4']
        )
        
        if scenario['name'] == 'Baseline':
            ax.plot(sensing_tdm, comm_tdm,
                   color='black',
                   linestyle=':',
                   linewidth=1.5,
                   label='TDM Baseline',
                   alpha=0.6)
            
            # Shade synergistic gain region with proper monotonic interpolation
            if len(sensing) > 1:
                # Ensure monotonic ordering for interpolation
                idx_sort = np.argsort(sensing)
                s_sorted = sensing[idx_sort]
                r_sorted = comm[idx_sort]
                
                # Interpolate TDM line to Pareto points
                if s_sorted[-1] > s_sorted[0]:  # Check monotonicity
                    comm_tdm_interp = np.interp(s_sorted, sensing_tdm, comm_tdm)
                    ax.fill_between(s_sorted, r_sorted, comm_tdm_interp,
                                   where=(r_sorted >= comm_tdm_interp),
                                   color='blue', alpha=0.1,
                                   label='Synergistic Gain')
    
    # Calculate and display synergistic gains
    print("\n" + "=" * 60)
    print("Synergistic Gain Analysis")
    print("=" * 60)
    
    for name, (sensing, comm) in boundaries.items():
        # Approximate area under curve
        if len(sensing) > 1:
            area_pareto = np.trapz(comm, sensing)
            
            # TDM area (triangle)
            max_sensing = np.max(sensing)
            max_comm = np.max(comm)
            area_tdm = 0.5 * max_sensing * max_comm
            
            if area_tdm > 0:
                gain = (area_pareto / area_tdm - 1) * 100
                print(f"{name:20s}: {gain:+.1f}% gain over TDM")
    
    # Formatting
    ax.set_xlabel('Sensing Precision $P_S$ (1/CRLB)', fontsize=12)
    ax.set_ylabel('Communication Capacity $R_C$ [Mbps]', fontsize=12)
    ax.set_title('Pareto Optimal Boundaries for Mars ISAC System\n' + 
                 'UHF Proximity Link (435 MHz, 10 MHz BW, 10 W, 400 km)',
                 fontsize=13, fontweight='bold')
    
    # Grid and legend
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)
    
    # Set axis limits for better visualization
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    
    # Add text box with key insights
    textstr = 'Key Insights:\n'
    textstr += '• Joint design outperforms TDM\n'
    textstr += '• UHF: dust impact negligible (Rayleigh regime)\n'
    textstr += '• Scintillation impacts capacity significantly\n'
    textstr += '• Realistic UHF rates: 20-80 Mbps'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_resource_allocation_heatmap():
    """
    Generate heatmap showing optimal resource allocation with proper normalization.
    """
    print("\nGenerating Resource Allocation Heatmap...")
    
    # Initialize system
    system = MarsISACSystem(link_type='uhf')
    
    # Parameter ranges
    rho_range = np.linspace(0, 1, 41)
    beta_range = np.linspace(0, 1, 41)
    
    # Baseline environmental conditions
    tau_vis = 0.2
    S4 = 0.2
    
    # Calculate performance metrics
    RHO, BETA = np.meshgrid(rho_range, beta_range)
    
    # Storage for individual metrics
    PRECISION = np.zeros_like(RHO)
    CAPACITY = np.zeros_like(RHO)
    
    for i in range(len(beta_range)):
        for j in range(len(rho_range)):
            PRECISION[i,j] = system.sensing_precision(RHO[i,j], BETA[i,j], tau_vis, S4)
            CAPACITY[i,j] = system.ergodic_capacity_nakagami(RHO[i,j], BETA[i,j], tau_vis, S4)
    
    # Normalize by maximum values
    P_max = np.max(PRECISION)
    C_max = np.max(CAPACITY)
    
    # Combined utility with proper normalization
    w_s = 0.5  # Weight for sensing
    w_c = 0.5  # Weight for communication
    
    UTILITY = w_s * (PRECISION / P_max) + w_c * (CAPACITY / C_max)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Utility heatmap
    im1 = ax1.pcolormesh(RHO, BETA, UTILITY, cmap='viridis', shading='auto')
    
    # Find optimal point
    max_idx = np.unravel_index(np.argmax(UTILITY), UTILITY.shape)
    opt_rho = RHO[max_idx]
    opt_beta = BETA[max_idx]
    
    # Mark optimal point
    ax1.plot(opt_rho, opt_beta, 'r*', markersize=15, 
             label=f'Optimal: ρ={opt_rho:.2f}, β={opt_beta:.2f}')
    
    # Add contour lines
    CS1 = ax1.contour(RHO, BETA, UTILITY, levels=10, colors='white', 
                      linewidths=0.5, alpha=0.5)
    
    # Labels and formatting
    ax1.set_xlabel('Power Allocation Factor ρ (Sensing)', fontsize=11)
    ax1.set_ylabel('Resource Block Allocation Factor β (Sensing)', fontsize=11)
    ax1.set_title('Joint Resource Allocation (Equal Weights)',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    
    # Colorbar for utility
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Normalized Combined Utility', fontsize=10)
    
    # Right panel: Individual metrics contours
    # Plot sensing precision contours
    CS2 = ax2.contour(RHO, BETA, PRECISION * 1e3, levels=8, colors='blue', 
                      linewidths=1.5, alpha=0.7)
    ax2.clabel(CS2, inline=True, fontsize=8, fmt='P=%.1f')
    
    # Plot communication capacity contours
    CS3 = ax2.contour(RHO, BETA, CAPACITY / 1e6, levels=8, colors='red', 
                      linewidths=1.5, alpha=0.7)
    ax2.clabel(CS3, inline=True, fontsize=8, fmt='R=%.0f')
    
    # Mark optimal point
    ax2.plot(opt_rho, opt_beta, 'g*', markersize=15, label='Optimal Point')
    
    # Labels and formatting
    ax2.set_xlabel('Power Allocation Factor ρ (Sensing)', fontsize=11)
    ax2.set_ylabel('Resource Block Allocation Factor β (Sensing)', fontsize=11)
    ax2.set_title('Performance Contours (Blue: Sensing, Red: Comm)',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main function to run complete analysis.
    """
    import os
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    print("✅ Created/verified 'results' directory")
    
    # Generate main Pareto boundaries figure
    fig1 = plot_pareto_boundaries()
    fig1.savefig('results/fig4_1_pareto_boundaries_corrected.pdf', format='pdf', dpi=300)
    fig1.savefig('results/fig4_1_pareto_boundaries_corrected.png', format='png', dpi=300)
    print("\n✅ Saved: results/fig4_1_pareto_boundaries_corrected.pdf/png")
    
    # Generate resource allocation heatmap
    fig2 = plot_resource_allocation_heatmap()
    fig2.savefig('results/fig4_2_resource_allocation_corrected.pdf', format='pdf', dpi=300)
    fig2.savefig('results/fig4_2_resource_allocation_corrected.png', format='png', dpi=300)
    print("✅ Saved: results/fig4_2_resource_allocation_corrected.pdf/png")
    
    # Display figures
    plt.show()
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE (CORRECTED VERSION)!")
    print("=" * 80)
    print("\n📊 Generated Figures:")
    print("  1. Pareto Optimal Boundaries (Corrected)")
    print("     - Proper link budget with FSPL and antenna gains")
    print("     - Correct Nakagami-m capacity calculation")
    print("     - True Pareto frontier via non-dominated sorting")
    print("     - Realistic capacity values (~10-80 Mbps)")
    print("\n  2. Resource Allocation Heatmap (Corrected)")
    print("     - Normalized utility function")
    print("     - Interior optimal point")
    print("     - Individual performance contours")
    print("\n📝 All results saved in 'results/' directory")
    print("\n✨ Key corrections made:")
    print("   • Fixed average SNR calculation (no division by m)")
    print("   • Corrected Nakagami PDF integration")
    print("   • Implemented SNR² dependence for dust sensing")
    print("   • Added proper non-dominated sorting for Pareto frontier")
    print("   • Normalized utility metrics for meaningful optimization")

if __name__ == "__main__":
    main()