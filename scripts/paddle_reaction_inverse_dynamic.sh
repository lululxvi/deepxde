set -e
export PYTHONPATH=./
mkdir -p reaction_inverse
DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/reaction_inverse.py > ./reaction_inverse/dynamic.log
