set -e
export PYTHONPATH=./
mkdir -p ode_2nd
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/ode_2nd.py > ./ode_2nd/dynamic.log
