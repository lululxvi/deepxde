set -e
export PYTHONPATH=./
mkdir -p fractional_Poisson_1d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/fractional_Poisson_1d.py > ./fractional_Poisson_1d/dynamic.log
