set -e
export PYTHONPATH=./
mkdir -p fractional_diffusion_1d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/fractional_diffusion_1d.py > ./fractional_diffusion_1d/dynamic.log
