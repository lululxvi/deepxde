set -e
export PYTHONPATH=./
mkdir -p wave_1d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/wave_1d.py > ./wave_1d/dynamic.log
