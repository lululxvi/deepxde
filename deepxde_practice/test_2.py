import tensorflow as tf
import numpy as np

print("=" * 50)
print("GradientTape — Building toward PINNs")
print("=" * 50)

# ── 1. Basic first derivative ────────────────────────────────
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**3 + 2*x
dy_dx = tape.gradient(y, x)
print(f"\n1. dy/dx at x=3: {dy_dx.numpy()}")
print(f"   Expected: 29.0  (3x² + 2 = 27 + 2)")

# ── 2. Second derivative (needed for ∂²T/∂x² in heat eq.) ───
x = tf.Variable(3.0)
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = x**4          # y = x⁴
    dy_dx = tape1.gradient(y, x)       # dy/dx  = 4x³
d2y_dx2 = tape2.gradient(dy_dx, x)    # d²y/dx² = 12x²
print(f"\n2. d²y/dx² at x=3: {d2y_dx2.numpy()}")
print(f"   Expected: 108.0  (12x² = 12×9)")

# ── 3. Gradient through a neural network (core PINN idea) ───
print("\n3. Gradient through a neural network:")

# Build a tiny neural network: input (x,t) → output u
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(1)
])

# A single point (x=0.5, t=0.2) — in real PINN this is thousands of points
xt = tf.constant([[0.5, 0.2]])   # shape (1, 2)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(xt)               # watch input (not a Variable, so must watch manually)
    u = model(xt)                # forward pass through network

# Compute ∂u/∂x and ∂u/∂t
grads   = tape.gradient(u, xt)  # shape (1, 2) — gradients w.r.t. both x and t
du_dx   = grads[:, 0]           # first column  = ∂u/∂x
du_dt   = grads[:, 1]           # second column = ∂u/∂t

print(f"   u(0.5, 0.2)  = {u.numpy()[0][0]:.4f}")
print(f"   ∂u/∂x        = {du_dx.numpy()[0]:.4f}")
print(f"   ∂u/∂t        = {du_dt.numpy()[0]:.4f}")
print("   (values are random — network not trained yet, that's fine)")

# ── 4. The PDE residual — corrected version ─────────────────
print("\n4. PDE residual for 1D heat equation (∂u/∂t - α∂²u/∂x² = 0):")
alpha = 0.4

xt = tf.Variable([[0.5, 0.2]])   # ← KEY FIX: use tf.Variable, not tf.constant
                                  # Variables are watched automatically by all tapes

with tf.GradientTape(persistent=True) as tape2:
    with tf.GradientTape(persistent=True) as tape1:
        u      = model(xt)
        du_dx  = tape1.gradient(u, xt)[:, 0:1]    # ∂u/∂x
        du_dt  = tape1.gradient(u, xt)[:, 1:2]    # ∂u/∂t

d2u_dx2 = tape2.gradient(du_dx, xt)[:, 0:1]       # ∂²u/∂x²

pde_residual = du_dt - alpha * d2u_dx2
print(f"   ∂u/∂t              = {du_dt.numpy()[0][0]:.4f}")
print(f"   α × ∂²u/∂x²       = {(alpha * d2u_dx2).numpy()[0][0]:.4f}")
print(f"   PDE residual       = {pde_residual.numpy()[0][0]:.4f}")
print("   (goal of training: drive this residual → 0.0)")

print("\n" + "=" * 50)
print("✓ GradientTape working correctly for PINN use")
print("=" * 50)

# cleanup
del tape1, tape2