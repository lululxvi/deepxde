set -e
export PYTHONPATH=./
mkdir -p Lorenz_inverse
DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/Lorenz_inverse.py > ./Lorenz_inverse/dynamic.log
