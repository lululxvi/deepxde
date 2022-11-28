set -e
export PYTHONPATH=./
mkdir -p diffusion_reaction
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/diffusion_reaction.py > ./diffusion_reaction/dynamic.log
