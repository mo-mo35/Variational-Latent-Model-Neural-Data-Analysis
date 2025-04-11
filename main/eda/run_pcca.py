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

# -------------------------------------------------------------------------
# Global definitions
# -------------------------------------------------------------------------
mm = np.matmul
inv = np.linalg.inv

# This folder is where results are saved
current_dir = os.path.dirname(os.path.abspath(__file__))
results_folder = os.path.join(current_dir, "..", "results")
os.makedirs(results_folder, exist_ok=True)
print(f"Saving results to folder: {results_folder}")

# -------------------------------------------------------------------------
# PCCA Class (Unchanged)
# -------------------------------------------------------------------------
class OptimizedPCCA:
    def __init__(self, n_components, max_iter=30, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        X1, X2 = X
        assert X1.shape[0] == X2.shape[0], "Both views must have the same number of samples"
        p1 = X1.shape[1]
        p2 = X2.shape[1]
        self.n_samples = X1.shape[0]
        X_joint = np.hstack((X1, X2)).T  # shape (p1+p2, n_samples)
        Lambda, Psi = self._init_params(p1, p2)
        XX_t = np.dot(X_joint, X_joint.T)
        prev_nll = float('inf')
        for i in range(self.max_iter):
            Lambda_new, Psi_new = self._em_step(X_joint, Lambda, Psi, XX_t)
            nll = self._neg_log_likelihood(X_joint, Lambda_new, Psi_new)
            if i > 0 and abs(prev_nll - nll) < self.tol:
                print(f"Converged after {i+1} iterations")
                break
            Lambda, Psi = Lambda_new, Psi_new
            prev_nll = nll
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
        Lambda1 = np.random.random((p1, k))
        Lambda2 = np.random.random((p2, k))
        Psi1_init = 0.5 * np.eye(p1)
        Psi2_init = 0.5 * np.eye(p2)
        Lambda = np.vstack((Lambda1, Lambda2))
        Psi = np.zeros((p1 + p2, p1 + p2))
        Psi[:p1, :p1] = Psi1_init
        Psi[p1:, p1:] = Psi2_init
        return Lambda, Psi

    def _neg_log_likelihood(self, X, Lambda, Psi):
        p, n = X.shape
        log_det_psi = np.log(np.linalg.det(Psi))
        Psi_inv = linalg.inv(Psi)
        trace_term = np.trace(Psi_inv.dot(X).dot(X.T))
        nll = 0.5 * (n * log_det_psi + trace_term)
        return nll

    def _em_step(self, X, Lambda, Psi, XX_t):
        p, n = X.shape
        k = self.n_components
        LP_term = np.dot(Lambda, Lambda.T) + Psi
        LP_inv = linalg.inv(LP_term)
        beta = np.dot(Lambda.T, LP_inv)
        Ez = np.dot(beta, X)
        I_minus_bL = np.eye(k) - np.dot(beta, Lambda)
        bXXb = np.dot(Ez, Ez.T)
        EzzT = n * I_minus_bL + bXXb
        Lambda_lterm = np.dot(X, Ez.T)
        Lambda_rterm_inv = linalg.inv(EzzT)
        Lambda_new = np.dot(Lambda_lterm, Lambda_rterm_inv)
        Psi_new = XX_t - np.dot(Lambda_new, np.dot(Ez, X.T))
        Psi_diag = np.diag(np.diag(Psi_new)) / n
        return Lambda_new, Psi_diag

    def E_z_given_x(self, Lambda, Psi, X):
        beta = np.dot(Lambda.T, linalg.inv(np.dot(Lambda, Lambda.T) + Psi))
        return np.dot(beta, X)

    def reconstruct(self):
        Z = self.E_z_given_x(self.Lambda, self.Psi, self.X)
        X1_rec = np.dot(self.Lambda1, Z).T
        X2_rec = np.dot(self.Lambda2, Z).T
        return X1_rec, X2_rec

    def sample(self, n):
        np.random.seed(1)
        Z = self.E_z_given_x(self.Lambda, self.Psi, self.X)
        m1 = np.dot(self.Lambda1, Z)
        m2 = np.dot(self.Lambda2, Z)
        p1, _ = self.Lambda1.shape
        p2, _ = self.Lambda2.shape
        X1 = np.zeros((n, p1))
        X2 = np.zeros((n, p2))
        for i in range(n):
            X1[i, :] = np.random.multivariate_normal(mean=m1[:, i], cov=self.Psi1)
            X2[i, :] = np.random.multivariate_normal(mean=m2[:, i], cov=self.Psi2)
        return X1, X2

# -------------------------------------------------------------------------
# Additional helper functions (unchanged)
# -------------------------------------------------------------------------
def calculate_r_squared(X_true, X_pred):
    n_dims = X_true.shape[1]
    total_ss = 0
    total_resid = 0
    for d in range(n_dims):
        x = X_true[:, d]
        y = X_pred[:, d]
        total_ss += np.sum((x - np.mean(x))**2)
        total_resid += np.sum((x - y)**2)
    return {"overall": 1 - total_resid/total_ss}

def run_pcca_components(n_components, X1, X2, scaler1=None, scaler2=None):
    n_min = min(X1.shape[0], X2.shape[0])
    X1 = X1[:n_min, :]
    X2 = X2[:n_min, :]
    print(f"Running PCCA with {n_components} components...")
    pcca = OptimizedPCCA(n_components=n_components, max_iter=30, tol=1e-4)
    pcca.fit([X1, X2])
    X1_rec, X2_rec = pcca.reconstruct()
    X1_sample, X2_sample = pcca.sample(n_min)
    rmse1 = np.sqrt(np.mean((X1 - X1_sample)**2))
    rmse2 = np.sqrt(np.mean((X2 - X2_sample)**2))
    total_rmse = rmse1 + rmse2
    r_squared1 = calculate_r_squared(X1, X1_rec)
    r_squared2 = calculate_r_squared(X2, X2_rec)
    avg_r2 = (r_squared1["overall"] + r_squared2["overall"]) / 2
    p1 = pcca.Lambda1.shape[0]
    X_joint = pcca.X
    X1_orig = X_joint[:p1, :].T
    X2_orig = X_joint[p1:, :].T
    X1_shared = X1_rec.copy()
    X2_shared = X2_rec.copy()
    X1_unique = X1_orig - X1_shared
    X2_unique = X2_orig - X2_shared
    results = {
        "rmse": (rmse1, rmse2, total_rmse),
        "r_squared": (r_squared1, r_squared2, avg_r2),
        "pcca_model": pcca,
        "decomposition": {
            "X1_shared": X1_shared,
            "X1_unique": X1_unique,
            "X2_shared": X2_shared,
            "X2_unique": X2_unique
        }
    }
    return results

def analyze_pcca_performance(X1, X2, max_components=6, scaler1=None, scaler2=None,
                             region1_label="Region1", region2_label="Region2", n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_pcca_components)(i, X1, X2, scaler1, scaler2)
        for i in range(1, max_components+1)
    )
    print(f"\nSummary of {region1_label}-{region2_label} pCCA Results:")
    for idx, res in enumerate(results, start=1):
        rmse1, rmse2, _ = res["rmse"]
        r1, r2, avg_r2 = res["r_squared"]
        print(f"  n_components={idx}: RMSE1={rmse1:.3f}, RMSE2={rmse2:.3f}, R²1={r1['overall']:.3f}, R²2={r2['overall']:.3f}, AvgR²={avg_r2:.3f}")
    return {"full_results": results}

# -------------------------------------------------------------------------
# New function: Overlay Decomposition Plot
# -------------------------------------------------------------------------
def plot_decomposition_overlay(X1_orig, X2_orig, shared1, shared2, event_label, results_folder, save_filename):
    """
    Overlays the reconstruction decomposition plots for Region1 and Region2 on a single figure.
    It displays, for the first feature/dimension:
      - The original signal for both regions,
      - The shared reconstruction for both regions,
      - The unique residual (Original - Shared) for each region.
    """
    n_points = min(50, X1_orig.shape[0], X2_orig.shape[0])
    t = np.linspace(-1.25, 1.25, n_points)
    
    unique1 = X1_orig[:n_points, 0] - shared1[:n_points, 0]
    unique2 = X2_orig[:n_points, 0] - shared2[:n_points, 0]
    
    plt.figure(figsize=(12, 7))
    # Region1
    plt.plot(t, X1_orig[:n_points, 0], 'k-', label="Region1 Original", linewidth=1.8)
    plt.plot(t, shared1[:n_points, 0], 'b--', label="Region1 Shared", linewidth=1.8)
    plt.plot(t, unique1, 'b:', label="Region1 Unique", linewidth=1.8)
    # Region2 (Note: use "gray" instead of "gray-" for a valid color specification)
    plt.plot(t, X2_orig[:n_points, 0], 'gray', label="Region2 Original", linewidth=1.8)
    plt.plot(t, shared2[:n_points, 0], 'r--', label="Region2 Shared", linewidth=1.8)
    plt.plot(t, unique2, 'r:', label="Region2 Unique", linewidth=1.8)
    
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Scaled Signal", fontsize=14)
    plt.title(f"{event_label} Reconstruction Decomposition (Overlay)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    out_path = os.path.join(results_folder, save_filename)
    plt.savefig(out_path)
    print(f"Overlay decomposition plot saved to: {out_path}")
    plt.show()

# -------------------------------------------------------------------------
# Other Plotting Functions (unchanged)
# -------------------------------------------------------------------------
def plot_shared_latent_scatter(pcca, latent_dim, event_name, results_folder, time_axis=None):
    Z = pcca.E_z_given_x(pcca.Lambda, pcca.Psi, pcca.X)
    r1 = np.dot(pcca.Lambda1[latent_dim:latent_dim+1, :], Z).flatten()
    r2 = np.dot(pcca.Lambda2[latent_dim:latent_dim+1, :], Z).flatten()
    
    plt.figure(figsize=(8,6))
    if time_axis is not None:
        sc = plt.scatter(r1, r2, c=time_axis, cmap="viridis", alpha=0.6)
        cbar = plt.colorbar(sc)
        cbar.set_label("Time")
    else:
        plt.scatter(r1, r2, alpha=0.6)
    corr = np.corrcoef(r1, r2)[0,1]
    plt.xlabel(f"Region1 latent {latent_dim+1}")
    plt.ylabel(f"Region2 latent {latent_dim+1}")
    plt.title(f"{event_name} Shared Latent {latent_dim+1}\nCorrelation={corr:.3f}")
    plt.grid(True)
    outname = os.path.join(results_folder, f"{event_name.lower()}_shared_latent_scatter_latent_{latent_dim+1}.png")
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
            mode='lines', line={'color':'blue','width':1}, name='Region1'))
        fig.add_trace(Scatter3d(
            x=X2[:, 0], y=X2[:, 1], z=X2[:, 2],
            mode='lines', line={'color':'orange','width':1}, name='Region2'))
        fig.update_layout(scene=dict(
                                xaxis_title="Latent Variable 1",
                                yaxis_title="Latent Variable 2",
                                zaxis_title="Latent Variable 3"),
                          width=900, height=700,
                          title=f"{event_label}: Latent Variables")
        outpath = os.path.join(results_folder, f"{event_label.lower()}_plot.html")
        fig.write_html(outpath)
        print(f"{event_label} 3D plot saved to: {outpath}")
        fig.show()
    else:
        plt.figure(figsize=(8,6))
        plt.plot(X1[:, 0], X1[:, 1], 'o-', color='blue', label='Region1')
        plt.plot(X2[:, 0], X2[:, 1], 's-', color='orange', label='Region2')
        plt.xlabel("Latent Variable 1")
        plt.ylabel("Latent Variable 2")
        plt.title(f"{event_label}: Latent Variables (2D)")
        plt.legend()
        outpath = os.path.join(results_folder, f"{event_label.lower()}_plot.png")
        plt.savefig(outpath)
        print(f"{event_label} 2D plot saved to: {outpath}")
        plt.show()

# -------------------------------------------------------------------------
# Main pCCA Analysis Helper Functions (unchanged)
# -------------------------------------------------------------------------
def run_pcca_components(n_components, X1, X2, scaler1=None, scaler2=None):
    n_min = min(X1.shape[0], X2.shape[0])
    X1 = X1[:n_min, :]
    X2 = X2[:n_min, :]
    print(f"Running PCCA with {n_components} components...")
    pcca = OptimizedPCCA(n_components=n_components, max_iter=30, tol=1e-4)
    pcca.fit([X1, X2])
    X1_rec, X2_rec = pcca.reconstruct()
    X1_sample, X2_sample = pcca.sample(n_min)
    rmse1 = np.sqrt(np.mean((X1 - X1_sample)**2))
    rmse2 = np.sqrt(np.mean((X2 - X2_sample)**2))
    total_rmse = rmse1 + rmse2
    r_squared1 = calculate_r_squared(X1, X1_rec)
    r_squared2 = calculate_r_squared(X2, X2_rec)
    avg_r2 = (r_squared1["overall"] + r_squared2["overall"]) / 2
    p1 = pcca.Lambda1.shape[0]
    X_joint = pcca.X
    X1_orig = X_joint[:p1, :].T
    X2_orig = X_joint[p1:, :].T
    X1_shared = X1_rec.copy()
    X2_shared = X2_rec.copy()
    X1_unique = X1_orig - X1_shared
    X2_unique = X2_orig - X2_shared
    results = {
        "rmse": (rmse1, rmse2, total_rmse),
        "r_squared": (r_squared1, r_squared2, avg_r2),
        "pcca_model": pcca,
        "decomposition": {
            "X1_shared": X1_shared,
            "X1_unique": X1_unique,
            "X2_shared": X2_shared,
            "X2_unique": X2_unique
        }
    }
    return results

def analyze_pcca_performance(X1, X2, max_components=6, scaler1=None, scaler2=None,
                             region1_label="Region1", region2_label="Region2", n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_pcca_components)(i, X1, X2, scaler1, scaler2)
        for i in range(1, max_components+1)
    )
    print(f"\nSummary of {region1_label}-{region2_label} pCCA Results:")
    for idx, res in enumerate(results, start=1):
        rmse1, rmse2, _ = res["rmse"]
        r1, r2, avg_r2 = res["r_squared"]
        print(f"  n_components={idx}: RMSE1={rmse1:.3f}, RMSE2={rmse2:.3f}, R²1={r1['overall']:.3f}, R²2={r2['overall']:.3f}, AvgR²={avg_r2:.3f}")
    return {"full_results": results}

# -------------------------------------------------------------------------
# MAIN CODE: Process Each Event
# -------------------------------------------------------------------------
with open("all_vlgp_models.pkl", "rb") as file:
    loaded_data = pickle.load(file)

session = list(loaded_data.keys())[0]
print(f"Processing session: {session}")

def process_event(event_name, data_dict):
    # Step1: Identify the two relevant regions
    regions = list(data_dict.keys())
    if len(regions) < 2:
        raise ValueError(f"{event_name} event must have at least 2 regions")
    r1, r2 = regions[0], regions[1]
    print(f"{event_name} event using regions: {r1} and {r2}")

    # Step2: Load data
    X1 = np.vstack([trial["mu"] for trial in data_dict[r1]["trials"]])
    X2 = np.vstack([trial["mu"] for trial in data_dict[r2]["trials"]])

    # Step3: Scale data
    scaler1 = StandardScaler()
    X1_scaled = scaler1.fit_transform(X1)
    scaler2 = StandardScaler()
    X2_scaled = scaler2.fit_transform(X2)

    # Step4: Analyze pCCA performance
    results_dict = analyze_pcca_performance(X1_scaled, X2_scaled,
                                            max_components=6,
                                            scaler1=scaler1, scaler2=scaler2,
                                            region1_label=r1, region2_label=r2, n_jobs=1)
    full_res = results_dict["full_results"]
    selected_index = 2  # use the model with 3 latent factors (0-indexed)
    decomp = full_res[selected_index]["decomposition"]
    pcca_model = full_res[selected_index]["pcca_model"]

    # Retrieve original joint signals from the model for overlay
    p1_dim = pcca_model.Lambda1.shape[0]
    X1_orig = pcca_model.X[:p1_dim, :].T
    X2_orig = pcca_model.X[p1_dim:, :].T

    # Step5: Plot event latents (2D or 3D)
    plot_event_latents(X1_scaled, X2_scaled, event_name, results_folder)

    # Step6: Call the new overlay decomposition plot for the event
    overlay_filename = f"{event_name.lower()}_overlay_decomposition.png"
    plot_decomposition_overlay(X1_orig, X2_orig, decomp["X1_shared"], decomp["X2_shared"],
                               event_name, results_folder, overlay_filename)

    # Step7: Plot shared latent scatter plots for each latent dimension
    time_axis = np.linspace(-1.25, 1.25, pcca_model.X.shape[1])
    for d in range(pcca_model.n_components):
        plot_shared_latent_scatter(pcca_model, latent_dim=d, event_name=event_name,
                                   results_folder=results_folder, time_axis=time_axis)

# Process each event
process_event("Reward", loaded_data[session]["reward"])
process_event("Movement", loaded_data[session]["movement"])
process_event("Stimulus", loaded_data[session]["stimulus"])
