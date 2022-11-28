set -e
export PYTHONPATH=./
mkdir -p Navier_Stokes_inverse
DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/Navier_Stokes_inverse.py > ./Navier_Stokes_inverse/dynamic.log
