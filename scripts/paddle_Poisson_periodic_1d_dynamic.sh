set -e
export PYTHONPATH=./
mkdir -p Poisson_periodic_1d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Poisson_periodic_1d.py > ./Poisson_periodic_1d/dynamic.log
