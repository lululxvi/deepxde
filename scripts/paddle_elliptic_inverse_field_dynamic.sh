set -e
export PYTHONPATH=./
mkdir -p elliptic_inverse_field
DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/elliptic_inverse_field.py > ./elliptic_inverse_field/dynamic.log
