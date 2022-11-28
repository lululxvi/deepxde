set -e
export PYTHONPATH=./
mkdir -p Helmholtz_Neumann_2d_hole
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Helmholtz_Neumann_2d_hole.py > ./Helmholtz_Neumann_2d_hole/dynamic.log
