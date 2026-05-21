#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-$HOME/envs/synthmorph-gpu}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found. Install it first: sudo apt install -y python3 python3-venv python3-pip"
  exit 1
fi

if ! python3 -m venv "$ENV_DIR" >/dev/null 2>&1; then
  echo "Could not create a venv. On Ubuntu, run: sudo apt install -y python3-venv"
  exit 1
fi

source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$SCRIPT_DIR/requirements_wsl_gpu.txt"

python - <<'PY'
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY

PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import collections
import inspect
import tensorflow as tf
import tensorflow.keras.losses as losses

if not hasattr(inspect, "getargspec"):
    ArgSpec = collections.namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec

if not hasattr(losses, "mean_absolute_error"):
    losses.mean_absolute_error = lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)
if not hasattr(losses, "mean_squared_error"):
    losses.mean_squared_error = lambda y_true, y_pred: tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)

import neurite
import voxelmorph

print("Neurite:", getattr(neurite, "__version__", "unknown"))
print("Voxelmorph:", getattr(voxelmorph, "__version__", "unknown"))
PY

echo "WSL SynthMorph GPU environment is ready: $ENV_DIR/bin/python"
