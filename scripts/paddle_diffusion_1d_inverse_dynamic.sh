set -e
export PYTHONPATH=./
mkdir -p diffusion_1d_inverse
DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/diffusion_1d_inverse.py > ./diffusion_1d_inverse/dynamic.log
