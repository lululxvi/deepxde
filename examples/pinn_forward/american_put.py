"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle

Black-Scholes PDE for American Put Option:
∂V/∂t + 0.5 * σ^2 * S^2 * ∂^2V/∂S^2 + r * S * ∂V/∂S - r * V = 0, for S > 0, t < T
With the early exercise constraint:
V(S, t) >= max(K - S, 0) for all S, t
And the complementarity condition:
(V(S, t) - max(K - S, 0)) * (∂V/∂t + 0.5 * σ^2 * S^2 * ∂^2V/∂S^2 + r * S * ∂V/∂S - r * V) = 0   
"""
import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import math 

# --- 1. Financial Parameters ---
r = 0.05
sigma = 0.2
K = 1.0 
# Anything too high above strike price will have zero payoff, so we can limit the domain to a few multiples of K
S_max = K * 4.0 
T = 1.0

# --- 2. PDE Definition (Fischer-Burmeister LCP) ---
def pde(x, y):
    s = x[:, 0:1]
    v = y
    
    dv_dtau = dde.grad.jacobian(y, x, i=0, j=1) 
    dv_ds = dde.grad.jacobian(y, x, i=0, j=0)
    dv_ds2 = dde.grad.hessian(y, x, i=0, j=0)
    
    # Black-Scholes Operator (Time reversed: tau = T - t)
    bs_op = -dv_dtau + 0.5 * (sigma**2) * (s**2) * dv_ds2 + r * s * dv_ds - r * v
    
    # Intrinsic Value (Payoff)
    payoff = dde.backend.relu(K - s)
    
    # LCP Constraints
    # 1. Soft penalty for L(V) > 0
    res_pde = dde.backend.relu(bs_op) 
    
    # 2. Soft penalty for V < Payoff
    penalty = dde.backend.relu(payoff - v)
    
    # 3. Fischer-Burmeister Complementarity
    # phi(a, b) = sqrt(a^2 + b^2) - (a + b) = 0
    a = v - payoff
    b = -bs_op 
    # Add 1e-8 for numerical stability of the derivative of sqrt at 0
    res_fb = torch.sqrt(a**2 + b**2 + 1e-8) - (a + b)
    
    return [res_pde, penalty, res_fb]

# --- 3. Geometry and Time Domain ---
geom = dde.geometry.Interval(0, S_max)
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Create anchor points dense around the strike price (S=1.0) 
# This forces the network to learn the kink perfectly.
S_anchors = np.random.uniform(0.5 * K, 1.5 * K, (1000, 1))
tau_anchors = np.random.uniform(0, T, (1000, 1))
anchors = np.hstack((S_anchors, tau_anchors))

# --- 4. Boundary and Initial Conditions ---
def initial_condition(x):
    s = x[:, 0:1]
    return np.maximum(K - s, 0)

IC = dde.icbc.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)

def low_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def high_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], S_max)

LB = dde.icbc.DirichletBC(geomtime, lambda x: K, low_boundary, component=0)
HB = dde.icbc.DirichletBC(geomtime, lambda x: 0, high_boundary, component=0)

# Compile Data
data = dde.data.TimePDE(
    geomtime, 
    pde, 
    [IC, LB, HB],
    num_domain=6000, 
    num_boundary=1500, 
    num_initial=1500,
    anchors=anchors # Pass the focus points
)

# --- 5. Network Architecture ---
# Increased width slightly to handle the sharper FB gradients
net = dde.nn.FNN([2] + [128] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# --- 6. Training ---
print("Stage 1: Training with Adam...")
# Weights: [res_pde, penalty, res_fb, IC, LB, HB]
# Heavily weight the initial/boundary conditions and the payoff penalty
model.compile("adam", lr=0.001, loss_weights=[1, 10, 5, 100, 100, 100])
model.train(iterations=12000)

print("Stage 2: Fine-tuning with L-BFGS and Adaptive Refinement...")
model.compile("L-BFGS")
checker = dde.callbacks.PDEPointResampler(period=1000)
losshistory, train_state = model.train(callbacks=[checker])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# --- 7. Ground Truth Evaluation ---
def binomial_american_put(S, K, T, r, sigma, N=2000):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    S_tree = S * d**(np.arange(N, -1, -1)) * u**(np.arange(0, N + 1, 1))
    V = np.maximum(K - S_tree, 0)
    
    for i in range(N - 1, -1, -1):
        V = (p * V[1:] + (1 - p) * V[:-1]) * np.exp(-r * dt)
        S_i = S * d**(np.arange(i, -1, -1)) * u**(np.arange(0, i + 1, 1))
        V = np.maximum(V, K - S_i) 
        
    return V[0]

S_eval = np.linspace(0.1, 2.0, 100).reshape(-1, 1) 
tau_eval = np.ones_like(S_eval) * T
X_test = np.hstack((S_eval, tau_eval))

y_pred = model.predict(X_test)

print("Calculating Binomial Ground Truth (this may take a moment)...")
y_true = [binomial_american_put(s[0], K, T, r, sigma) for s in S_eval]

plt.figure(figsize=(10, 6))
plt.plot(S_eval, y_true, 'r-', label='Ground Truth (Binomial Tree)', linewidth=2)
plt.plot(S_eval, y_pred, 'b--', label='PINN Prediction', linewidth=2)
plt.axvline(K, color='black', linestyle=':', label='Strike Price (K)')
plt.xlabel('Stock Price (S)')
plt.ylabel('Option Value (V)')
plt.title('American Put Option: PINN vs. Ground Truth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("american_put_comparison.png")
plt.show()

error_l2 = np.linalg.norm(y_true - y_pred.flatten()) / np.linalg.norm(y_true)
print(f"Relative L2 Error: {error_l2:.4e}")