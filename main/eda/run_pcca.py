import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import time
from joblib import Parallel, delayed
import plotly.graph_objects as go

# Some parts of this pCCA implementation are from https://github.com/gwgundersen/ml/ by Gregory Gundersen
mm = np.matmul
inv = np.linalg.inv

# Optimize matrix operations by using scipy.linalg where possible (faster than np.linalg)
# and caching repeated computations
class OptimizedPCCA:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        """
        Initialize the optimized probabilistic CCA model.
        
        Parameters:
        - n_components: The dimensionality of the latent variable
        - max_iter: Maximum number of EM iterations
        - tol: Convergence tolerance for negative log-likelihood
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X):
        """
        Fit the probabilistic CCA model using Expectation-Maximization with early stopping.
        
        Parameters:
        - X: A list or tuple containing two datasets X1 and X2
        """
        X1, X2 = X
        n1, p1 = X1.shape
        n2, p2 = X2.shape
        assert n1 == n2, "Both views must have the same number of samples"
        n = n1
        p = p1 + p2
        k = self.n_components
        
        # Stack data horizontally and transpose
        X_joint = np.hstack(X).T
        assert X_joint.shape == (p, n)
        
        # Initialize parameters
        Lambda, Psi = self._init_params(p1, p2)
        
        # Precompute X * X.T once
        XX_t = np.dot(X_joint, X_joint.T)
        
        # For tracking convergence
        nlls = []
        prev_nll = float('inf')
        
        # EM iterations with early stopping
        for i in range(self.max_iter):
            start_time = time.time()
            
            # EM step with optimized matrix operations
            Lambda_new, Psi_new = self._em_step(X_joint, Lambda, Psi, XX_t)
            
            # Calculate negative log-likelihood
            nll = self._neg_log_likelihood(X_joint, Lambda, Psi)
            nlls.append(nll)
            
            # Check for convergence
            if i > 0 and abs(prev_nll - nll) < self.tol:
                print(f"Converged after {i+1} iterations")
                break
                
            prev_nll = nll
            Lambda = Lambda_new
            Psi = Psi_new
            
            iter_time = time.time() - start_time
            if (i+1) % 10 == 0:
                print(f"Iteration {i+1}/{self.max_iter} completed in {iter_time:.2f}s - NLL: {nll:.4f}")
        
        # Store model components
        self.X = X_joint
        self.Lambda = Lambda
        self.Lambda1 = Lambda[:p1, :]
        self.Lambda2 = Lambda[p1:, :]
        self.Psi = Psi
        self.Psi1 = Psi[:p1, :p1]
        self.Psi2 = Psi[p1:, p1:]
        self.nlls = nlls
        
        return self
        
    def sample(self, n):
        """
        Sample from the fitted probabilistic CCA model.
        
        Parameters:
        - n: The number of samples
        
        Returns:
        - Two views of n samples each
        """
        # Seed for reproducibility
        np.random.seed(1)
        
        # Calculate latent variable expectation
        Z = self.E_z_given_x(self.Lambda, self.Psi, self.X)
        
        # Project through the loading matrices
        m1 = np.dot(self.Lambda1, Z)
        m2 = np.dot(self.Lambda2, Z)
        
        # Get dimensions
        p1, _ = self.Lambda1.shape
        p2, _ = self.Lambda2.shape
        
        # Initialize output arrays
        X1 = np.zeros((n, p1))
        X2 = np.zeros((n, p2))
        
        # Generate samples
        for i in range(n):
            X1[i, :] = np.random.multivariate_normal(mean=m1[:, i], cov=self.Psi1)
            X2[i, :] = np.random.multivariate_normal(mean=m2[:, i], cov=self.Psi2)
        
        return X1, X2
    
    def reconstruct(self):
        """
        Reconstruct the data from the latent variables (without adding noise).
        This is useful for calculating R-squared.
        
        Returns:
        - Reconstructed views X1 and X2
        """
        # Calculate latent variable expectation
        Z = self.E_z_given_x(self.Lambda, self.Psi, self.X)
        
        # Project through loading matrices (without adding noise)
        X1_rec = np.dot(self.Lambda1, Z).T
        X2_rec = np.dot(self.Lambda2, Z).T
        
        return X1_rec, X2_rec
        
    def _em_step(self, X, Lambda, Psi, XX_t):
        """
        Optimized EM step that reuses precomputed values and uses efficient matrix operations.
        """
        p, n = X.shape
        k = self.n_components
        
        # Cached computation for (LL^T + Psi)^-1
        LP_term = np.dot(Lambda, Lambda.T) + Psi
        LP_inv = linalg.inv(LP_term)
        
        # E[z|x]
        beta = np.dot(Lambda.T, LP_inv)
        Ez = np.dot(beta, X)
        
        # E[zz^T|x]
        I_minus_bL = np.eye(k) - np.dot(beta, Lambda)
        bXXb = np.dot(Ez, Ez.T)
        EzzT = n * I_minus_bL + bXXb
        
        # Update Lambda
        Lambda_lterm = np.dot(X, Ez.T)
        Lambda_rterm_inv = linalg.inv(EzzT)
        Lambda_new = np.dot(Lambda_lterm, Lambda_rterm_inv)
        
        # Update Psi
        Psi_new = XX_t - np.dot(Lambda_new, np.dot(Ez, X.T))
        Psi_diag = np.diag(np.diag(Psi_new)) / n
        
        return Lambda_new, Psi_diag
    
    def _init_params(self, p1, p2):
        """
        Initialize the model parameters.
        """
        k = self.n_components
        
        # Random initialization with fixed seed for reproducibility
        np.random.seed(42)
        
        # Initialize Lambda with PCA to get a better starting point
        Lambda1 = np.random.random((p1, k))
        Lambda2 = np.random.random((p2, k))
        
        # Initialize Psi as diagonal matrices with small values
        Psi1_init = 0.5 * np.eye(p1)
        Psi2_init = 0.5 * np.eye(p2)
        
        # Combine for joint model
        Lambda = np.vstack((Lambda1, Lambda2))
        
        # Block-diagonal Psi
        Psi = np.zeros((p1 + p2, p1 + p2))
        Psi[:p1, :p1] = Psi1_init
        Psi[p1:, p1:] = Psi2_init
        
        return Lambda, Psi
    
    def _neg_log_likelihood(self, X, Lambda, Psi):
        """
        Compute negative log-likelihood with optimized matrix operations.
        """
        p, n = X.shape
        
        # Compute the determinant of Psi
        log_det_psi = np.log(np.linalg.det(Psi))
        
        # Invert Psi once
        Psi_inv = linalg.inv(Psi)
        
        # Calculate the negative log-likelihood efficiently
        LP_term = np.dot(Lambda, Lambda.T) + Psi
        LP_inv = linalg.inv(LP_term)
        
        # Compute trace term
        trace_term = np.trace(np.dot(Psi_inv, np.dot(X, X.T)))
        
        # Additional terms
        beta = np.dot(Lambda.T, LP_inv)
        Ez = np.dot(beta, X)
        reconstruction = np.dot(Lambda, Ez)
        reconstruction_error = np.sum((X - reconstruction) ** 2)
        
        # Simplified NLL calculation
        nll = 0.5 * (n * log_det_psi + trace_term)
        
        return nll

    def E_z_given_x(self, L, P, X):
        beta = mm(L.T, inv(mm(L, L.T) + P))
        return mm(beta, X)

    def E_zzT_given_x(self, L, P, X, k):
        beta = mm(L.T, inv(mm(L, L.T) + P))
        bX   = mm(beta, X)
        if len(X.shape) == 2:
            # See here for details:
            # https://stackoverflow.com/questions/48498662/
            _, N = X.shape
            bXXb = np.einsum('ib,ob->io', bX, bX)
            return N * (np.eye(k) - mm(beta, L)) + bXXb
        else:
            bXXb = np.outer(bX, bX.T)
            return np.eye(k) - mm(beta, L) + bXXb

# Calculate R-squared for each dimension and overall
def calculate_r_squared(X_true, X_pred):
    """
    Calculate R-squared for each dimension and overall.
    
    Parameters:
    - X_true: True data matrix
    - X_pred: Predicted data matrix
    
    Returns:
    - Dictionary with dimension-wise and overall R-squared values
    """
    # Number of dimensions
    n_dims = X_true.shape[1]
    
    # Initialize results dictionary
    r_squared = {'overall': 0, 'dimensions': []}
    
    # Calculate R-squared for each dimension
    total_ss = 0
    total_residual_ss = 0
    
    for dim in range(n_dims):
        y_true = X_true[:, dim]
        y_pred = X_pred[:, dim]
        
        # Sum of squares total
        ss_total = np.sum((y_true - np.mean(y_true))**2)
        
        # Sum of squares residual
        ss_residual = np.sum((y_true - y_pred)**2)
        
        # R-squared for this dimension
        r_sq_dim = 1 - (ss_residual / ss_total)
        r_squared['dimensions'].append(r_sq_dim)
        
        # Accumulate for overall R-squared
        total_ss += ss_total
        total_residual_ss += ss_residual
    
    # Calculate overall R-squared
    r_squared['overall'] = 1 - (total_residual_ss / total_ss)
    
    return r_squared

# Function to run PCCA with specific number of components
def run_pcca_components(n_components, X1, X2):
    """Run PCCA with specified number of components and return RMSEs and R-squared values"""
    print(f"Running PCCA with {n_components} components...")
    
    # Time the execution
    start_time = time.time()
    
    # Create and fit PCCA model
    pcca = OptimizedPCCA(n_components=n_components, max_iter=30, tol=1e-4)
    pcca.fit([X1, X2])
    
    # Generate reconstructions (without noise for R-squared calculation)
    X1_rec, X2_rec = pcca.reconstruct()
    
    # Generate samples (with noise for RMSE calculation)
    X1_sample, X2_sample = pcca.sample(X1.shape[0])
    
    # Calculate RMSE
    rmse1 = np.sqrt(np.mean((X1 - X1_sample) ** 2))
    rmse2 = np.sqrt(np.mean((X2 - X2_sample) ** 2))
    total_rmse = rmse1 + rmse2
    
    # Calculate R-squared
    r_squared1 = calculate_r_squared(X1, X1_rec)
    r_squared2 = calculate_r_squared(X2, X2_rec)
    
    # Calculate average R-squared across both datasets
    avg_r_squared = (r_squared1['overall'] + r_squared2['overall']) / 2
    
    elapsed_time = time.time() - start_time
    print(f"  Components: {n_components}, Time: {elapsed_time:.2f}s")
    print(f"  RMSE1: {rmse1:.4f}, RMSE2: {rmse2:.4f}, Total RMSE: {total_rmse:.4f}")
    print(f"  R² Region1: {r_squared1['overall']:.4f}, R² Region2: {r_squared2['overall']:.4f}, Avg R²: {avg_r_squared:.4f}")
    
    results = {
        'rmse': (rmse1, rmse2, total_rmse),
        'r_squared': (r_squared1, r_squared2, avg_r_squared),
        'time': elapsed_time
    }
    
    return results

# Main function to run analysis with parallel processing
def analyze_pcca_performance(X1, X2, max_components=6, n_jobs=-1):
    """
    Analyze PCCA performance for different numbers of components
    
    Parameters:
    - X1, X2: Input data matrices
    - max_components: Maximum number of components to try
    - n_jobs: Number of parallel jobs (-1 for all CPUs)
    """
    # Run PCCA for different numbers of components in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_pcca_components)(i, X1, X2) 
        for i in range(1, max_components + 1)
    )
    
    # Organize results
    rmse_dict = {f"latent{i+1}": [results[i]['rmse']] for i in range(max_components)}
    r_squared_dict = {f"latent{i+1}": [results[i]['r_squared']] for i in range(max_components)}
    
    # Print summary
    print("\nSummary of PCCA Results:")
    print("-----------------------")
    print("\nRMSE Results:")
    for k, v in rmse_dict.items():
        print(f"{k}: RMSE Region 1 = {v[0][0]:.4f}, RMSE Region 2 = {v[0][1]:.4f}, Total = {v[0][2]:.4f}")
    
    print("\nR-squared Results:")
    for k, v in r_squared_dict.items():
        print(f"{k}: R² Region 1 = {v[0][0]['overall']:.4f}, R² Region 2 = {v[0][1]['overall']:.4f}, Avg R² = {v[0][2]:.4f}")
    
    # Plot results
    plot_pcca_results(rmse_dict, r_squared_dict, max_components)
    
    return {'rmse': rmse_dict, 'r_squared': r_squared_dict}

# Function to plot PCCA results with both RMSE and R-squared
def plot_pcca_results(rmse_dict, r_squared_dict, max_components):
    """
    Plot both RMSE and R-squared results for PCCA models
    
    Parameters:
    - rmse_dict: Dictionary containing RMSE results
    - r_squared_dict: Dictionary containing R-squared results
    - max_components: Maximum number of components evaluated
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 11))
    
    # Create x-ticks
    x_ticks = list(range(1, max_components + 1))
    
    # ===== RMSE Plot (Left) =====
    # Extract RMSEs
    rmse_region1 = [v[0][0] for v in rmse_dict.values()]
    rmse_region2 = [v[0][1] for v in rmse_dict.values()]
    total_rmse = [v[0][2] for v in rmse_dict.values()]
    
    # Plot RMSE lines with markers
    ax1.plot(x_ticks, rmse_region1, 'o-', label='SCdg', linewidth=2)
    ax1.plot(x_ticks, rmse_region2, 's-', label='SCiw', linewidth=2)
    ax1.plot(x_ticks, total_rmse, '^--', label='Total RMSE', linewidth=2, alpha=0.7)
    
    # Add grid and labels for RMSE
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Number of Latent Variables', fontsize=16)
    ax1.set_ylabel('RMSE', fontsize=16)
    ax1.set_xticks(x_ticks)
    ax1.tick_params(axis='both', labelsize=14)
    
    # Add legend and title for RMSE
    ax1.legend(fontsize=16)
    ax1.set_title('RMSE by Number of Latent Variables', fontsize=18)
    
    # Annotate minimum points for RMSE
    min_idx1 = np.argmin(rmse_region1)
    min_idx2 = np.argmin(rmse_region2)
    min_idx_total = np.argmin(total_rmse)
    
    ax1.annotate(f'Min: {rmse_region1[min_idx1]:.4f}',
                xy=(x_ticks[min_idx1], rmse_region1[min_idx1]),
                xytext=(10, -20), textcoords='offset points', fontsize=14,
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    ax1.annotate(f'Min: {rmse_region2[min_idx2]:.4f}',
                xy=(x_ticks[min_idx2], rmse_region2[min_idx2]),
                xytext=(10, 20), textcoords='offset points', fontsize=14,
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # ===== R-squared Plot (Right) =====
    # Extract R-squared values
    r2_region1 = [v[0][0]['overall'] for v in r_squared_dict.values()]
    r2_region2 = [v[0][1]['overall'] for v in r_squared_dict.values()]
    r2_avg = [v[0][2] for v in r_squared_dict.values()]
    
    # Plot R-squared lines with different markers
    ax2.plot(x_ticks, r2_region1, 'o-', label='SCdg', linewidth=2)
    ax2.plot(x_ticks, r2_region2, 's-', label='SCiw', linewidth=2)
    ax2.plot(x_ticks, r2_avg, '^--', label='Average R²', linewidth=2, alpha=0.7)
    
    # Add grid and labels for R-squared
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Number of Latent Variables', fontsize=16)
    ax2.set_ylabel('R-squared', fontsize=16)
    ax2.set_xticks(x_ticks)
    ax2.tick_params(axis='both', labelsize=14)
    
    # Set y-axis limits for R-squared (typically between 0 and 1)
    ax2.set_ylim([-0.1, 1.1])
    
    # Add legend and title for R-squared
    ax2.legend(fontsize=16, loc='upper left')
    ax2.set_title('R-squared by Number of Latent Variables', fontsize=18)
     
    # Annotate maximum points for R-squared
    max_idx1 = np.argmax(r2_region1)
    max_idx2 = np.argmax(r2_region2)
    max_idx_avg = np.argmax(r2_avg)
    
    ax2.annotate(f'Max: {r2_region1[max_idx1]:.4f}',
                xy=(x_ticks[max_idx1], r2_region1[max_idx1]),
                xytext=(10, -20), textcoords='offset points', fontsize=14,
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    ax2.annotate(f'Max: {r2_region2[max_idx2]:.4f}',
                xy=(x_ticks[max_idx2], r2_region2[max_idx2]),
                xytext=(10, 20), textcoords='offset points', fontsize=14,
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Add a shared suptitle
    plt.suptitle('PCCA Performance Metrics', fontsize=28)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Show plot
    plt.show()
    fig.savefig('pCCA_RMSE_R2_plot.png')
    return fig



with open('all_vlgp_models.pkl', 'rb') as file:
    loaded_data = pickle.load(file)


#SCiw group 1 stacking the trials: dimension goes from latents x observations --> latents x (trials*observations)
sciw = loaded_data[list(loaded_data.keys())[0]]['reward']['SCiw']['trials']
X1 = np.vstack([sciw[i]['mu'] for i in np.arange(len(sciw))])
scaler = StandardScaler()
X1 = scaler.fit_transform(X1)
#SCdg group 2
scdg = loaded_data[list(loaded_data.keys())[0]]['reward']['SCdg']['trials']
X2 = np.vstack([scdg[i]['mu'] for i in np.arange(len(scdg))])
scaler = StandardScaler()
X2 = scaler.fit_transform(X2)

rmse_results = analyze_pcca_performance(X1, X2, max_components=6)


#SCiw movement plot

#left
sciw = loaded_data[list(loaded_data.keys())[0]]['movement']['SCiw']['trials']
X1 = np.vstack([sciw[i]['mu'] for i in np.arange(len(sciw)) if sciw[i]['condition'] == 'left'])
scaler = StandardScaler()
X1 = scaler.fit_transform(X1)

#right
X2 = np.vstack([sciw[i]['mu'] for i in np.arange(len(sciw)) if sciw[i]['condition'] == 'right'])
scaler = StandardScaler()
X2 = scaler.fit_transform(X2)

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=X1[:, 0], y=X1[:, 1], z=X1[:, 2], mode='lines',
                        line={'color':'blue', 'width':1}, legendgroup='left', name='Wheel Turned Left', showlegend=True))
fig.add_trace(go.Scatter3d(x=X2[:, 0], y=X2[:, 1], z=X2[:, 2], mode='lines',
                        line={'color':'orange', 'width':1}, legendgroup='right', name='Wheel Turned Right', showlegend=True))
fig.update_layout(scene = dict(
                    xaxis_title='Latent Variable 1',
                    yaxis_title='Latent Variable 2',
                    zaxis_title='Latent Variable 3'),
                    width=1000, height=1000, title='Latent Variables'
                    )

fig.show()
fig.write_html("sciw_movement_plot.html")

#SCdg stimulus plot

#left
scdg = loaded_data[list(loaded_data.keys())[0]]['stimulus']['SCdg']['trials']
X1 = np.vstack([scdg[i]['mu'] for i in np.arange(len(scdg)) if scdg[i]['condition'] == 'left'])
scaler = StandardScaler()
X1 = scaler.fit_transform(X1)

#right
X2 = np.vstack([scdg[i]['mu'] for i in np.arange(len(scdg)) if scdg[i]['condition'] == 'right'])
scaler = StandardScaler()
X2 = scaler.fit_transform(X2)
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=X1[:, 0], y=X1[:, 1], z=X1[:, 2], mode='lines',
                        line={'color':'blue', 'width':1}, legendgroup='left', name='Wheel Turned Left', showlegend=True))
fig.add_trace(go.Scatter3d(x=X2[:, 0], y=X2[:, 1], z=X2[:, 2], mode='lines',
                        line={'color':'orange', 'width':1}, legendgroup='right', name='Wheel Turned Right', showlegend=True))
fig.update_layout(scene = dict(
                    xaxis_title='Latent Variable 1',
                    yaxis_title='Latent Variable 2',
                    zaxis_title='Latent Variable 3'),
                    width=1000, height=1000, title='Latent Variables'
                    )

fig.show()
fig.write_html("scdg_stimulus_plot.html")
