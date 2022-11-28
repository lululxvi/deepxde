set -e
export PYTHONPATH=./
mkdir -p diffusion_1d_resample
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/diffusion_1d_resample.py > ./diffusion_1d_resample/dynamic.log
