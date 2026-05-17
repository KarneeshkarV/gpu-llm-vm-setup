# _env.sh — sourced by run.sh / chat.sh / setup.sh.
#
# Activates the venv and exports LD_LIBRARY_PATH so the prebuilt CUDA wheel
# can find libcudart.so.12 / libcublas (the driver-only AMI has no CUDA
# toolkit; we ship the runtime libs as pip wheels).
#
# NOTE: the `nvidia` package is a namespace package — `nvidia.__file__` is
# None — so we locate the lib dirs via sysconfig's purelib, not __file__.

_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"

# shellcheck disable=SC1091
source "${_ENV_DIR}/.venv/bin/activate"

_SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="${_SP}/nvidia/cuda_runtime/lib:${_SP}/nvidia/cublas/lib:${_SP}/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
