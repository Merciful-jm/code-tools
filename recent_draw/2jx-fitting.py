
"""
Curie-Weiss refit with χ₀ anchored to physically
reasonable (Pascal + Pauli) values.

* 1/χ′ = a T + b  linearisation
* χ′ = χ - χ₀,   χ₀ chosen to maximise R²
* Returns C = 1/a, T₀ = -b C
* Produces one PDF per element.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path

# -------------------------
# Raw χ(T) data
# -------------------------
T = np.array([116, 195, 290, 580, 1160, 2000, 3000, 4000, 6000, 8000], dtype=float)

data = {
    "Cu(o)": np.array([0.46386, 0.47766, 0.47421, 0.45754, 0.4358,
                       0.41748, 0.39481, 0.36398, 0.29701, 0.23989]),
    "Cu(p)": np.array([0.40542, 0.4141, 0.40396, 0.40124, 0.39794,
                       0.37629, 0.33988, 0.30179, 0.23999, 0.19639]),
    "Cu(b)": np.array([0.11886, 0.11687, 0.11832, 0.11869, 0.11834,
                       0.12614, 0.15439, 0.18048, 0.19346, 0.17761]),
    "213-Cu(p)": np.array([0.66034, 0.64716, 0.62953, 0.61577, 0.5822,
                           0.48641, 0.38874, 0.3239, 0.24626, 0.19906]),
}

# -------------------------
# Pascal + Pauli prior estimates (rough)
# Reference based on observed magnitudes, same units as original χ
# -------------------------
pascal_estimates = {
    "Cu(o)": -0.03,
    "Cu(p)": -0.03,
    "Cu(b)": -0.02,
    "213-Cu(p)": -0.02,
}

results = {}

# -------------------------
# Model
# -------------------------
def curie_weiss(T, chi0, C, T0):
    return chi0 + C / (T - T0)

# -------------------------
# Main loop: For each temperature threshold, exhaustively search χ₀ to maximize linearity of 1/χ′ vs T
# -------------------------
thresholds = [1000, 2000, 3000, 4000]

for element, chi in data.items():
    results[element] = {}
    
    for threshold in thresholds:
        dir_path = Path(f"refit_plots>{threshold}")
        dir_path.mkdir(exist_ok=True)
        
        mask = T >= threshold
        T_fit = T[mask]
        chi_fit = chi[mask]
        
        chi0_grid = np.linspace(pascal_estimates[element] - 0.1,
                                pascal_estimates[element] + 0.1, 4001)
        
        best_r2, best = -np.inf, None
        
        for chi0 in chi0_grid:
            chi_corr = chi_fit - chi0
            if np.any(chi_corr <= 0):
                continue  # Need χ′>0 for physical meaning
            y_inv = 1.0 / chi_corr
            reg = LinearRegression().fit(T_fit.reshape(-1, 1), y_inv)
            r2 = reg.score(T_fit.reshape(-1, 1), y_inv)
            if r2 > best_r2:
                best_r2 = r2
                best = (chi0, reg.coef_[0], reg.intercept_)
        
        chi0_opt, a_opt, b_opt = best
        C_opt = 1.0 / a_opt
        T0_opt = -b_opt * C_opt
        
        results[element][threshold] = {
            'chi0': chi0_opt, 'C': C_opt, 'T0': T0_opt, 'R2': best_r2
        }
        
        # Fitting curve
        T_smooth = np.linspace(T.min(), T.max(), 400)
        chi_fit_curve = curie_weiss(T_smooth, chi0_opt, C_opt, T0_opt)
        
        # -------------------------
        # Plotting
        # -------------------------
        plt.figure(figsize=(7, 4.5))
        plt.scatter(T, chi, color="tab:orange", label="All Data")
        plt.scatter(T_fit, chi_fit, color="tab:red", marker="x", s=60, label=f"Fitted Data (≥{threshold} K)")
        plt.plot(T_smooth, chi_fit_curve, color="tab:red", linestyle="--", label=f"Fit (≥{threshold} K)")
        plt.xlabel("T (K)")
        plt.ylabel("χ")
        plt.title(f"{element} : Curie-Weiss refit ≥{threshold} K")
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Add fitted parameters as text annotations
        textstr = f'Refit ≥{threshold}K:\nχ₀ = {chi0_opt:.4f}\nC = {C_opt:.2f}\nT₀ = {T0_opt:.1f} K\nR² = {best_r2:.5f}'
        
        plt.text(1.05, 0.5, textstr, transform=plt.gca().transAxes, 
                 fontsize=9, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))
        
        out = dir_path / f"{element.replace('(','_').replace(')','_')}_refit.pdf"
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()

# -------------------------
# Plot all thresholds for each element in a single figure
# -------------------------
combined_dir = Path("combined_plots")
combined_dir.mkdir(exist_ok=True)

colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple']
T_smooth_full = np.linspace(T.min(), T.max(), 400)

for element, chi in data.items():
    plt.figure(figsize=(10, 6))
    
    # Plot all data points (put at bottom layer)
    plt.scatter(T, chi, color="tab:orange", s=80, label="All Data", zorder=1)
    
    # Plot fits for each threshold
    for i, threshold in enumerate(thresholds):
        if threshold in results[element]:
            params = results[element][threshold]
            chi0_opt = params['chi0']
            C_opt = params['C']
            T0_opt = params['T0']
            R2 = params['R2']
            
            # Fitting curve
            chi_fit_curve = curie_weiss(T_smooth_full, chi0_opt, C_opt, T0_opt)
            
            # Plot fitted data points
            mask = T >= threshold
            T_fit = T[mask]
            chi_fit = chi[mask]
            plt.scatter(T_fit, chi_fit, color=colors[i], marker='x', s=60, 
                       label=f"Fitted Data (≥{threshold} K)", zorder=5)
            
            # Plot fitting curve
            plt.plot(T_smooth_full, chi_fit_curve, color=colors[i], linestyle='--', 
                    linewidth=2, label=f"Fit ≥{threshold}K (R²={R2:.4f})", zorder=3)
    
    plt.xlabel("T (K)", fontsize=12)
    plt.ylabel("χ", fontsize=12)
    plt.title(f"{element} : Curie-Weiss refit - All temperature ranges", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add parameter table as text (positioned in upper right corner to avoid blocking curves)
    textstr = ""
    for i, threshold in enumerate(thresholds):
        if threshold in results[element]:
            params = results[element][threshold]
            textstr += f"≥{threshold}K: χ₀={params['chi0']:.4f}, C={params['C']:.2f}, T₀={params['T0']:.1f}K, R²={params['R2']:.4f}\n"
    
    plt.text(0.98, 0.98, textstr, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    out = combined_dir / f"{element.replace('(','_').replace(')','_')}_combined.pdf"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------
# Print results
# -------------------------
print("Element | Threshold | χ₀ | C | T₀ | R²")
for e, p in results.items():
    for threshold, vals in p.items():
        print(f"{e:10s} {threshold:5d} {vals['chi0']:7.4f} {vals['C']:10.2f} {vals['T0']:8.1f} {vals['R2']:7.5f}")
