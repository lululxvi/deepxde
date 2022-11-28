set -e
export PYTHONPATH=./
mkdir -p Poisson_Neumann_1d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Poisson_Neumann_1d.py > ./Poisson_Neumann_1d/dynamic.log
