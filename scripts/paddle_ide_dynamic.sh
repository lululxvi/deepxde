set -e
export PYTHONPATH=./
mkdir -p ide
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/ide.py > ./ide/dynamic.log
