set -e
export PYTHONPATH=./
mkdir -p ode_system
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/ode_system.py > ./ode_system/dynamic.log
