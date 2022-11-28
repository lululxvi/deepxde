set -e
export PYTHONPATH=./
mkdir -p fractional_Poisson_3d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/fractional_Poisson_3d.py > ./fractional_Poisson_3d/dynamic.log
