set -e
export PYTHONPATH=./
mkdir -p Poisson_Dirichlet_1d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Poisson_Dirichlet_1d.py > ./Poisson_Dirichlet_1d/dynamic.log
