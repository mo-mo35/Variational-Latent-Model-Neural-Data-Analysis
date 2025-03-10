# imports
import os
from os import path as op
import numpy as np
from scipy import stats
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
import vlgp
from vlgp import util, simulation

# simulated data generation
# Set dimensions and simulation parameters
K1 = 500  # Number of observations for Group 1
K2 = 800  # Number of observations for Group 2
K = K1 + K2  # Total number of observations
D = 6  # Total latent dimensions
T = 250  # Number of time points
t = np.linspace(0, 2, T)  # Time intervals

d_s = 2  # Shared latent dimension
d_1 = 2  # Independent latent dimension for group 1
d_2 = 2  # Independent latent dimension for group 2

rho = 1.0  # Scale for GP kernel
l = 2.0  # Length scale for GP kernel
nu = 0.1  # Noise variance for observations

# Set factor loadings
A_1 = np.random.randn(K1, d_1)
A_2 = np.random.randn(K2, d_2)
A_s1 = np.random.randn(K1, d_s)
A_s2 = np.random.randn(K2, d_s)

A = np.block([[A_s1, A_1, np.zeros((K1, d_2))],
              [A_s2, np.zeros((K2, d_1)), A_2]])  # Group 2

def kernel_function(t1, t2, rho, l):
    """Squared exponential kernel."""
    dist_sq = cdist(t1.reshape(-1, 1), t2.reshape(-1, 1), metric='sqeuclidean')
    return rho * np.exp(-dist_sq / (2 * l ** 2))

K_t = kernel_function(t, t, rho, l)

z_shared = np.random.multivariate_normal(np.zeros(T), K_t, size=d_s).T
z_1 = np.random.multivariate_normal(np.zeros(T), K_t, size=d_1).T
z_2 = np.random.multivariate_normal(np.zeros(T), K_t, size=d_2).T

Z = np.hstack([z_shared, z_1, z_2])

# trial setup
ntrial = 10  # Number of trials
nbin = 250  # Number of bins per trial to match T=250 as previously set
dim = 6  # Number latent dimensions
Z_cut = Z[:(Z.shape[0] // nbin) * nbin]
trials = [{'ID': i, 'y': Z_cut[i * nbin: (i + 1) * nbin].reshape(nbin, dim)} for i in range(Z_cut.shape[0] // nbin)]

#model 
fit = vlgp.fit(
    trials,  
    n_factors=3,  # dimensionality of latent process
    max_iter=20,  # maximum number of iterations
    min_iter=10  # minimum number of iterations
)

# validation and visualizations
trials = fit['trials']  # extract trials
for i in range(len(trials)):
    trial = trials[i]
    x = trial['y']  # observed data
    mu = trial['mu']  # posterior latent

    # Calculate the projection matrix from latent space to observed space
    W, _, _, _ = np.linalg.lstsq(mu, x, rcond=None)
    mu_proj = mu @ W 
    
    # Plotting
    plt.figure(figsize=(20, 10))
    for j in range(x.shape[1]):
        offset = 2 * j
        plt.plot(x[:, j] + offset, 'b', label='Observed' if j == 0 else "")
        plt.plot(mu_proj[:, j] + offset, 'r', label='Projected' if j == 0 else "")
    plt.legend()
    plt.show()
    plt.close()

# RMSE
rmses = [] 

for i, trial in enumerate(trials):
    x = trial['y']  # observed data
    mu = trial['mu']  # posterior latent

    # Calculate the projection matrix from latent space to observed space
    W, _, _, _ = np.linalg.lstsq(mu, x, rcond=None)
    mu_proj = mu @ W  
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((x - mu_proj) ** 2))
    rmses.append(rmse)

# correlation Coefficients
def calculate_correlation(cor, pred):
    correlation_matrix = np.corrcoef(cor.T, pred.T)
    num_variables = cor.shape[1]
    return np.diag(correlation_matrix[num_variables:, :num_variables])

correlations = [calculate_correlation(trial['y'], mu @ W) for trial in trials]