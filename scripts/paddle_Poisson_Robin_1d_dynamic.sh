set -e
export PYTHONPATH=./
mkdir -p Poisson_Robin_1d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Poisson_Robin_1d.py > ./Poisson_Robin_1d/dynamic.log
