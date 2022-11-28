set -e
export PYTHONPATH=./
mkdir -p Poisson_multiscale_1d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Poisson_multiscale_1d.py > ./Poisson_multiscale_1d/dynamic.log
