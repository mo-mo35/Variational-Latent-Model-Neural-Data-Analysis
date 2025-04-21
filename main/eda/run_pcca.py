#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import time
from joblib import Parallel, delayed
import plotly.graph_objects as go
import shutil
from sklearn.decomposition import PCA

# -------------------------------------------------------------------------
# Global definitions
# -------------------------------------------------------------------------
mm = np.matmul
inv = np.linalg.inv

current_dir = os.path.dirname(os.path.abspath(__file__))
results_folder = os.path.join(current_dir, "..", "results")
os.makedirs(results_folder, exist_ok=True)
print(f"Saving results to folder: {results_folder}")

# containers to collect metrics across events
all_rmse_results = {}
all_r2_results = {}

# -------------------------------------------------------------------------
# Optimized PCCA implementation
# -------------------------------------------------------------------------
class OptimizedPCCA:
    def __init__(self, n_components, max_iter=30, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        X1, X2 = X
        assert X1.shape[0] == X2.shape[0], "Both views must have the same number of samples"
        p1, p2 = X1.shape[1], X2.shape[1]
        self.n_samples = X1.shape[0]
        X_joint = np.hstack((X1, X2)).T
        Lambda, Psi = self._init_params(p1, p2)
        XX_t = X_joint @ X_joint.T
        prev_nll = float('inf')
        for i in range(self.max_iter):
            Lambda_new, Psi_new = self._em_step(X_joint, Lambda, Psi, XX_t)
            nll = self._neg_log_likelihood(X_joint, Lambda_new, Psi_new)
            if i > 0 and abs(prev_nll - nll) < self.tol:
                print(f"Converged after {i+1} iterations")
                break
            Lambda, Psi, prev_nll = Lambda_new, Psi_new, nll
            if (i+1) % 10 == 0:
                print(f"Iteration {i+1} - NLL: {nll:.4f}")
        self.X = X_joint
        self.Lambda = Lambda
        self.Lambda1 = Lambda[:p1, :]
        self.Lambda2 = Lambda[p1:, :]
        self.Psi = Psi
        self.Psi1 = Psi[:p1, :p1]
        self.Psi2 = Psi[p1:, p1:]
        return self

    def _init_params(self, p1, p2):
        k = self.n_components
        np.random.seed(42)
        L1 = np.random.random((p1, k))
        L2 = np.random.random((p2, k))
        Psi = np.zeros((p1+p2, p1+p2))
        Psi[:p1, :p1] = 0.5 * np.eye(p1)
        Psi[p1:, p1:] = 0.5 * np.eye(p2)
        return np.vstack((L1, L2)), Psi

    def _neg_log_likelihood(self, X, Lambda, Psi):
        p, n = X.shape
        logdet = np.log(np.linalg.det(Psi))
        Psi_inv = linalg.inv(Psi)
        trace_term = np.trace(Psi_inv @ X @ X.T)
        return 0.5 * (n * logdet + trace_term)

    def _em_step(self, X, Lambda, Psi, XX_t):
        p, n = X.shape
        k = self.n_components
        M = Lambda @ Lambda.T + Psi
        invM = linalg.inv(M)
        beta = Lambda.T @ invM
        Ez = beta @ X
        I_bL = np.eye(k) - beta @ Lambda
        Ezz = n * I_bL + Ez @ Ez.T
        Lambda_new = (X @ Ez.T) @ linalg.inv(Ezz)
        Psi_new = XX_t - Lambda_new @ Ez @ X.T
        Psi_diag = np.diag(np.diag(Psi_new)) / n
        return Lambda_new, Psi_diag

    def E_z_given_x(self, Lambda, Psi, X):
        beta = Lambda.T @ linalg.inv(Lambda @ Lambda.T + Psi)
        return beta @ X

    def reconstruct(self):
        Z = self.E_z_given_x(self.Lambda, self.Psi, self.X)
        return (self.Lambda1 @ Z).T, (self.Lambda2 @ Z).T

    def sample(self, n):
        np.random.seed(1)
        Z = self.E_z_given_x(self.Lambda, self.Psi, self.X)
        m1 = self.Lambda1 @ Z
        m2 = self.Lambda2 @ Z
        p1, p2 = self.Lambda1.shape[0], self.Lambda2.shape[0]
        X1 = np.array([np.random.multivariate_normal(m1[:, i], self.Psi1) for i in range(n)])
        X2 = np.array([np.random.multivariate_normal(m2[:, i], self.Psi2) for i in range(n)])
        return X1, X2

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def calculate_r_squared(X_true, X_pred):
    ss_tot = np.sum((X_true - X_true.mean(axis=0))**2)
    ss_res = np.sum((X_true - X_pred)**2)
    return {"overall": 1 - ss_res / ss_tot}

def run_pcca_components(n_components, X1, X2, scaler1=None, scaler2=None):
    n = min(X1.shape[0], X2.shape[0])
    X1, X2 = X1[:n], X2[:n]
    print(f"Running PCCA with {n_components} components...")
    pcca = OptimizedPCCA(n_components, max_iter=30, tol=1e-4).fit([X1, X2])
    X1_rec, X2_rec = pcca.reconstruct()
    X1_samp, X2_samp = pcca.sample(n)
    rmse1 = np.sqrt(np.mean((X1 - X1_samp)**2))
    rmse2 = np.sqrt(np.mean((X2 - X2_samp)**2))
    avg_r2 = np.mean([
        calculate_r_squared(X1, X1_rec)["overall"],
        calculate_r_squared(X2, X2_rec)["overall"]
    ])
    p1 = pcca.Lambda1.shape[0]
    joint = pcca.X
    return {
        "rmse": (rmse1, rmse2, rmse1 + rmse2),
        "r_squared": (
            calculate_r_squared(X1, X1_rec),
            calculate_r_squared(X2, X2_rec),
            avg_r2
        ),
        "pcca_model": pcca,
        "decomposition": {
            "X1_shared": X1_rec,
            "X2_shared": X2_rec,
            "X1_unique": joint[:p1].T - X1_rec,
            "X2_unique": joint[p1:].T - X2_rec
        }
    }

def analyze_pcca_performance(X1, X2, max_components=6, scaler1=None, scaler2=None,
                              region1_label="Region1", region2_label="Region2", n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_pcca_components)(k, X1, X2, scaler1, scaler2)
        for k in range(1, max_components+1)
    )
    print(f"\nSummary of {region1_label}-{region2_label} pCCA Results:")
    for i, res in enumerate(results, start=1):
        rm1, rm2, _ = res["rmse"]
        r1, r2, avg = res["r_squared"]
        print(f"  n_components={i}: RMSE1={rm1:.3f}, RMSE2={rm2:.3f}, "
              f"R²1={r1['overall']:.3f}, R²2={r2['overall']:.3f}, AvgR²={avg:.3f}")
    return {"full_results": results}

# -------------------------------------------------------------------------
# Plot decomposition & latents
# -------------------------------------------------------------------------
def plot_decomposition_overlay(X1_orig, X2_orig, shared1, shared2, event_label,
                               results_folder, save_filename):
    n_points = min(50, X1_orig.shape[0], X2_orig.shape[0])
    t = np.linspace(-1.25, 1.25, n_points)
    unique1 = X1_orig[:n_points,0] - shared1[:n_points,0]
    unique2 = X2_orig[:n_points,0] - shared2[:n_points,0]

    plt.figure(figsize=(12,7))
    plt.plot(t, X1_orig[:n_points,0], 'k-', label="Region1 Original", linewidth=1.8)
    plt.plot(t, X2_orig[:n_points,0], color='gray', label="Region2 Original", linewidth=1.8)
    plt.plot(t, shared1[:n_points,0], 'b--', label="Region1 Shared", linewidth=1.8)
    plt.plot(t, shared2[:n_points,0], 'r--', label="Region2 Shared", linewidth=1.8)
    plt.text(t[n_points//2], shared1[n_points//2,0]+0.05, "Extracted Shared (R1)", color='blue', fontsize=12)
    plt.text(t[n_points//2], shared2[n_points//2,0]-0.05, "Extracted Shared (R2)", color='red', fontsize=12)
    plt.plot(t, unique1, 'b:', label="Region1 Unique", linewidth=1.8)
    plt.plot(t, unique2, 'r:', label="Region2 Unique", linewidth=1.8)
    plt.text(t[n_points//4], unique1[n_points//4]+0.05, "Unique (R1)", color='blue', fontsize=12)
    plt.text(t[n_points//4], unique2[n_points//4]-0.05, "Unique (R2)", color='red', fontsize=12)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Scaled Signal", fontsize=14)
    plt.title(f"{event_label} Reconstruction Decomposition (Overlay)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    out_path = os.path.join(results_folder, save_filename)
    plt.savefig(out_path)
    print(f"Overlay decomposition plot saved to: {out_path}")
    plt.show()

def plot_shared_latent_scatter(pcca, latent_dim, event_name,
                               results_folder, time_axis=None):
    Z = pcca.E_z_given_x(pcca.Lambda, pcca.Psi, pcca.X)
    r1 = (pcca.Lambda1[latent_dim:latent_dim+1] @ Z).flatten()
    r2 = (pcca.Lambda2[latent_dim:latent_dim+1] @ Z).flatten()

    plt.figure(figsize=(8,6))
    if time_axis is not None:
        sc = plt.scatter(r1, r2, c=time_axis, cmap="viridis", alpha=0.6)
        plt.colorbar(sc, label="Time")
    else:
        plt.scatter(r1, r2, alpha=0.6)
    corr = np.corrcoef(r1, r2)[0,1]
    plt.xlabel(f"Region1 latent {latent_dim+1}", fontsize=12)
    plt.ylabel(f"Region2 latent {latent_dim+1}", fontsize=12)
    plt.title(f"{event_name} Shared Latent {latent_dim+1}\nCorrelation = {corr:.3f}", fontsize=14)
    plt.annotate("Strong shared structure",
                 xy=(r1.mean(), r2.mean()),
                 xytext=(r1.mean()+0.1, r2.mean()+0.1),
                 arrowprops=dict(facecolor='black', arrowstyle="->"),
                 fontsize=12)
    plt.grid(True)
    outname = os.path.join(results_folder,
                           f"{event_name.lower()}_shared_latent_scatter_latent_{latent_dim+1}.png")
    plt.savefig(outname)
    print(f"Shared latent scatter plot saved to {outname}")
    plt.show()

def plot_event_latents(X1, X2, event_label, results_folder):
    from plotly.graph_objects import Figure, Scatter3d
    min_dims = min(X1.shape[1], X2.shape[1])
    if min_dims >= 3:
        fig = Figure()
        fig.add_trace(Scatter3d(
            x=X1[:, 0], y=X1[:, 1], z=X1[:, 2],
            mode='lines', line={'color':'blue','width':2}, name='Region1'))
        fig.add_trace(Scatter3d(
            x=X2[:, 0], y=X2[:, 1], z=X2[:, 2],
            mode='lines', line={'color':'orange','width':2}, name='Region2'))
        fig.update_layout(scene=dict(
                            xaxis_title="Latent Variable 1",
                            yaxis_title="Latent Variable 2",
                            zaxis_title="Latent Variable 3"),
                          width=900, height=700,
                          title=f"{event_label}: 3D Latent Variables\n(Smooth Latent Trajectories)")
        outpath = os.path.join(results_folder, f"{event_label.lower()}_plot.html")
        fig.write_html(outpath)
        print(f"{event_label} 3D plot saved to: {outpath}")
        fig.show()
    else:
        plt.figure(figsize=(8,6))
        plt.plot(X1[:, 0], X1[:, 1], 'o-', color='blue', label='Region1')
        plt.plot(X2[:, 0], X2[:, 1], 's-', color='orange', label='Region2')
        plt.xlabel("Latent Variable 1", fontsize=12)
        plt.ylabel("Latent Variable 2", fontsize=12)
        plt.title(f"{event_label}: 2D Latent Variables", fontsize=14)
        plt.legend(fontsize=12)
        outpath = os.path.join(results_folder, f"{event_label.lower()}_plot.png")
        plt.savefig(outpath)
        print(f"{event_label} 2D plot saved to: {outpath}")
        plt.show()

'''def plot_pcca_results(rmse_dict, r_squared_dict, max_components):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 11))
    x_ticks = list(range(1, max_components + 1))

    # RMSE
    rmse_region1 = [v[0] for v in rmse_dict.values()]
    rmse_region2 = [v[1] for v in rmse_dict.values()]
    total_rmse   = [v[2] for v in rmse_dict.values()]

    ax1.plot(x_ticks, rmse_region1, 'o-', label='SCdg', linewidth=2)
    ax1.plot(x_ticks, rmse_region2, 's-', label='SCiw', linewidth=2)
    ax1.plot(x_ticks, total_rmse,  '^--', label='Total RMSE', linewidth=2, alpha=0.7)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Number of Latent Variables', fontsize=16)
    ax1.set_ylabel('RMSE', fontsize=16)
    ax1.set_xticks(x_ticks)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.legend(fontsize=16)
    ax1.set_title('RMSE by Number of Latent Variables', fontsize=18)

    # Annotate minima
    min1 = np.argmin(rmse_region1)
    min2 = np.argmin(rmse_region2)
    ax1.annotate(f'Min: {rmse_region1[min1]:.4f}',
                 xy=(x_ticks[min1], rmse_region1[min1]),
                 xytext=(10, -20), textcoords='offset points', fontsize=14,
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    ax1.annotate(f'Min: {rmse_region2[min2]:.4f}',
                 xy=(x_ticks[min2], rmse_region2[min2]),
                 xytext=(10,  20), textcoords='offset points', fontsize=14,
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    # R²
    r2_region1 = [v[0]['overall'] for v in r_squared_dict.values()]
    r2_region2 = [v[1]['overall'] for v in r_squared_dict.values()]
    r2_avg     = [v[2]            for v in r_squared_dict.values()]

    ax2.plot(x_ticks, r2_region1, 'o-', label='SCdg', linewidth=2)
    ax2.plot(x_ticks, r2_region2, 's-', label='SCiw', linewidth=2)
    ax2.plot(x_ticks, r2_avg,     '^--', label='Average R²', linewidth=2, alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Number of Latent Variables', fontsize=16)
    ax2.set_ylabel('R-squared', fontsize=16)
    ax2.set_xticks(x_ticks)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_ylim([-0.1, 1.1])
    ax2.legend(fontsize=16, loc='upper left')
    ax2.set_title('R-squared by Number of Latent Variables', fontsize=18)

    # Annotate maxima
    max1 = np.argmax(r2_region1)
    max2 = np.argmax(r2_region2)
    ax2.annotate(f'Max: {r2_region1[max1]:.4f}',
                 xy=(x_ticks[max1], r2_region1[max1]),
                 xytext=(10, -20), textcoords='offset points', fontsize=14,
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    ax2.annotate(f'Max: {r2_region2[max2]:.4f}',
                 xy=(x_ticks[max2], r2_region2[max2]),
                 xytext=(10,  20), textcoords='offset points', fontsize=14,
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    plt.suptitle('PCCA Performance Metrics', fontsize=28)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.savefig(os.path.join(results_folder, 'pCCA_RMSE_R2_plot.png'))
    plt.show()
    return fig'''

# -------------------------------------------------------------------------
# PCA baseline and comparison
# -------------------------------------------------------------------------
def run_pca_baseline(X1, X2, n_components=3):
    pca1 = PCA(n_components=n_components)
    pca2 = PCA(n_components=n_components)
    Z1 = pca1.fit_transform(X1)
    Z2 = pca2.fit_transform(X2)
    latent_corr = [np.corrcoef(Z1[:,i], Z2[:,i])[0,1] for i in range(n_components)]
    X1_rec = pca1.inverse_transform(Z1)
    X2_rec = pca2.inverse_transform(Z2)
    rmse1 = np.sqrt(np.mean((X1 - X1_rec)**2))
    rmse2 = np.sqrt(np.mean((X2 - X2_rec)**2))
    def compute_r2(X, Xrec):
        ss_res = np.sum((X - Xrec)**2)
        ss_tot = np.sum((X - X.mean(axis=0))**2)
        return 1 - ss_res/ss_tot
    r2_1 = compute_r2(X1,   X1_rec)
    r2_2 = compute_r2(X2,   X2_rec)
    avg_r2 = (r2_1 + r2_2) / 2
    return {'rmse': (rmse1, rmse2), 'r2': (r2_1, r2_2, avg_r2), 'latent_corr': latent_corr}

def compare_pipelines(event_name, data_dict, n_components=6):
    regions = list(data_dict.keys())
    if len(regions) < 2:
        raise ValueError(f"{event_name} event must have at least 2 regions")
    r1, r2 = regions[0], regions[1]
    print(f"\nEvent: {event_name} using regions: {r1} and {r2}")

    X1 = np.vstack([trial["mu"] for trial in data_dict[r1]["trials"]])
    X2 = np.vstack([trial["mu"] for trial in data_dict[r2]["trials"]])
    X1_s = StandardScaler().fit_transform(X1)
    X2_s = StandardScaler().fit_transform(X2)

    # Run PCCA pipeline
    perf = analyze_pcca_performance(
        X1_s, X2_s,
        max_components=n_components,
        scaler1=None, scaler2=None,
        region1_label=r1, region2_label=r2,
        n_jobs=1
    )
    full_results = perf["full_results"]
    selected = full_results[n_components-1]
    rm1_p, rm2_p, _ = selected["rmse"]
    r1_p, r2_p, avg_p = selected["r_squared"]

    # PCA baseline
    baseline = run_pca_baseline(X1_s, X2_s)

    print(f"\nComparison for event: {event_name}")
    print("VLGP->pCCA Pipeline:")
    print(f"  RMSE Region1: {rm1_p:.3f}, Region2: {rm2_p:.3f}")
    print(f"  R² Region1: {r1_p['overall']:.3f}, Region2: {r2_p['overall']:.3f}, Avg R²: {avg_p:.3f}")

    print("\nPCA Baseline (applied on VLGP latents):")
    rm1_b, rm2_b = baseline['rmse']
    r21_b, r22_b, avg_b = baseline['r2']
    print(f"  RMSE Region1: {rm1_b:.3f}, Region2: {rm2_b:.3f}")
    print(f"  R² Region1: {r21_b:.3f}, Region2: {r22_b:.3f}, Avg R²: {avg_b:.3f}")
    print(f"  Latent correlations (per component): {baseline['latent_corr']}")

    # Bar chart comparison (unchanged)
    labels = ['Region1 RMSE', 'Region2 RMSE', 'Region1 R²', 'Region2 R²']
    pcca_vals = [rm1_p, rm2_p, r1_p['overall'], r2_p['overall']]
    pca_vals  = [rm1_b, rm2_b, r21_b, r22_b]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width/2, pcca_vals, width, label='VLGP->pCCA')
    ax.bar(x + width/2, pca_vals,  width, label='PCA Baseline')
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title(f'Comparison of Pipelines for {event_name}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(pcca_vals):
        ax.text(x[i] - width/2, v + 0.01, f"{v:.2f}", ha='center', fontsize=12, fontweight='bold', color='blue')
    for i, v in enumerate(pca_vals):
        ax.text(x[i] + width/2, v + 0.01, f"{v:.2f}", ha='center', fontsize=12, fontweight='bold', color='red')
    save_path = os.path.join(results_folder, f"comparison_{event_name.lower()}.png")
    plt.savefig(save_path)
    print(f"Comparison plot saved to: {save_path}")
    plt.show()

    #Plot RMSE & R² across components
    rmse_dict = {i+1: full_results[i]["rmse"] for i in range(len(full_results))}
    r2_dict   = {i+1: full_results[i]["r_squared"] for i in range(len(full_results))}

    # collect for all-events plot
    global all_rmse_results, all_r2_results
    all_rmse_results[event_name] = rmse_dict
    all_r2_results[event_name] = r2_dict


# -------------------------------------------------------------------------
# Main event processing 
# -------------------------------------------------------------------------
def process_event(event_name, data_dict):
    regions = list(data_dict.keys())
    if len(regions) < 2:
        raise ValueError(f"{event_name} event must have at least 2 regions")
    r1, r2 = regions[0], regions[1]
    print(f"{event_name} event using regions: {r1} and {r2}")

    X1 = np.vstack([trial["mu"] for trial in data_dict[r1]["trials"]])
    X2 = np.vstack([trial["mu"] for trial in data_dict[r2]["trials"]])
    X1_s = StandardScaler().fit_transform(X1)
    X2_s = StandardScaler().fit_transform(X2)

    results = analyze_pcca_performance(X1_s, X2_s,
                                       max_components=6,
                                       scaler1=None, scaler2=None,
                                       region1_label=r1, region2_label=r2,
                                       n_jobs=1)
    full = results["full_results"]
    sel = full[2]  # index 2 = 3 components
    pcca = sel["pcca_model"]
    dec  = sel["decomposition"]

    p1 = pcca.Lambda1.shape[0]
    X1_orig = pcca.X[:p1].T
    X2_orig = pcca.X[p1:].T

    plot_event_latents(X1_s, X2_s, event_name, results_folder)
    plot_decomposition_overlay(X1_orig, X2_orig,
                               dec["X1_shared"], dec["X2_shared"],
                               event_name, results_folder,
                               f"{event_name.lower()}_overlay_decomposition.png")

    time_axis = np.linspace(-1.25, 1.25, pcca.X.shape[1])
    for d in range(pcca.n_components):
        plot_shared_latent_scatter(pcca, d, event_name, results_folder, time_axis)

    
    for region in (r1, r2):
        trials = data_dict[region]["trials"]
        X_left  = np.vstack([t["mu"] for t in trials if t.get("condition")=="left"])
        X_right = np.vstack([t["mu"] for t in trials if t.get("condition")=="right"])
        Xl = StandardScaler().fit_transform(X_left)
        Xr = StandardScaler().fit_transform(X_right)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=Xl[:,0], y=Xl[:,1], z=Xl[:,2],
            mode='lines', line={'color':'blue','width':1},
            name=f'{region} Left'))
        fig.add_trace(go.Scatter3d(
            x=Xr[:,0], y=Xr[:,1], z=Xr[:,2],
            mode='lines', line={'color':'orange','width':1},
            name=f'{region} Right'))
        fig.update_layout(
            scene=dict(xaxis_title='Latent Variable 1',
                       yaxis_title='Latent Variable 2',
                       zaxis_title='Latent Variable 3'),
            width=1000, height=1000,
            title=f'{region} {event_name}: Left vs Right Latent Trajectories'
        )
        outpath = os.path.join(results_folder,
                    f"{region.lower()}_{event_name.lower()}_left_right.html")
        fig.write_html(outpath)
        print(f"{region} left/right plot saved to: {outpath}")
        fig.show()

# -------------------------------------------------------------------------
# Plot all events’ RMSE & R² side by side
# -------------------------------------------------------------------------
def plot_all_events_metrics(all_rmse, all_r2, results_folder):
    events = list(all_rmse.keys())
    n_events = len(events)
    fig, axes = plt.subplots(n_events, 2, figsize=(18, 6 * n_events))
    for i, evt in enumerate(events):
        rmse_dict = all_rmse[evt]
        r2_dict   = all_r2[evt]
        x = list(rmse_dict.keys())

        # left: RMSE
        ax_rmse = axes[i,0] if n_events>1 else axes[0]
        ax_rmse.plot(x, [v[0] for v in rmse_dict.values()], 'o-', label='Region1', linewidth=2)
        ax_rmse.plot(x, [v[1] for v in rmse_dict.values()], 's-', label='Region2', linewidth=2)
        ax_rmse.set_title(f'{evt}: RMSE vs # Latent Vars', fontsize=16)
        ax_rmse.set_xlabel('Components', fontsize=14)
        ax_rmse.set_ylabel('RMSE', fontsize=14)
        ax_rmse.grid(True, linestyle='--', alpha=0.7)
        ax_rmse.legend(fontsize=12)

        # right: R²
        ax_r2 = axes[i,1] if n_events>1 else axes[1]
        ax_r2.plot(x, [v[0]['overall'] for v in r2_dict.values()], 'o-', label='Region1', linewidth=2)
        ax_r2.plot(x, [v[1]['overall'] for v in r2_dict.values()], 's-', label='Region2', linewidth=2)
        ax_r2.set_title(f'{evt}: R² vs # Latent Vars', fontsize=16)
        ax_r2.set_xlabel('Components', fontsize=14)
        ax_r2.set_ylabel('R²', fontsize=14)
        ax_r2.set_ylim([-0.1,1.1])
        ax_r2.grid(True, linestyle='--', alpha=0.7)
        ax_r2.legend(fontsize=12)

    plt.tight_layout()
    outpath = os.path.join(results_folder, 'all_events_pcca_metrics.png')
    fig.savefig(outpath)
    print(f"All-events RMSE & R² summary plot saved to: {outpath}")
    plt.show()

# -------------------------------------------------------------------------
# Main execution: dynamic over all events
# -------------------------------------------------------------------------
if __name__ == "__main__":
    with open("all_vlgp_models.pkl", "rb") as file:
        loaded_data = pickle.load(file)
    session = list(loaded_data.keys())[0]
    print(f"Processing session: {session}")
    
    for event_name, event_data in loaded_data[session].items():
        compare_pipelines(event_name.capitalize(), event_data)
        process_event(event_name.capitalize(), event_data)

    # finally, plot all side-by-side
    plot_all_events_metrics(all_rmse_results, all_r2_results, results_folder)
