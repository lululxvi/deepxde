set -e
export PYTHONPATH=./
mkdir -p fractional_Poisson_1d_inverse
DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/fractional_Poisson_1d_inverse.py > ./fractional_Poisson_1d_inverse/dynamic.log