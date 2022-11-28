set -e
export PYTHONPATH=./
mkdir -p fractional_Poisson_2d_inverse
DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/fractional_Poisson_2d_inverse.py > ./fractional_Poisson_2d_inverse/dynamic.log
