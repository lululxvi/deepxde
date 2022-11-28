set -e
export PYTHONPATH=./
mkdir -p Lotka_Volterra
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Lotka_Volterra.py > ./Lotka_Volterra/dynamic.log
