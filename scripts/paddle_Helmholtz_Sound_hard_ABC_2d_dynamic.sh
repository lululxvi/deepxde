set -e
export PYTHONPATH=./
mkdir -p Helmholtz_Sound_hard_ABC_2d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Helmholtz_Sound_hard_ABC_2d.py > ./Helmholtz_Sound_hard_ABC_2d/dynamic.log
