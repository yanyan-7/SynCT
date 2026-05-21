# SynCT SynthMorph WSL GPU Backend

SynCT can keep the 3D Slicer GUI on Windows while running SynthMorph registration in a WSL2 Linux Python environment with NVIDIA GPU support.

## 1. Install WSL2

Run in Windows PowerShell:

```powershell
wsl --install
wsl --update
```

This installs the default Ubuntu distribution. On some Windows/WSL builds the
distribution name `Ubuntu-22.04` is not available, or `wsl --list --online` may
fail if Windows cannot reach the online WSL distribution list.

If you want to pick a distribution explicitly, first run:

```powershell
wsl --list --online
```

Then install one of the names shown there, for example:

```powershell
wsl --install -d Ubuntu
```

If `wsl --list --online` reports `WININET_E_CANNOT_CONNECT`, install Ubuntu from
the Microsoft Store instead:

1. Open Microsoft Store.
2. Search for `Ubuntu`.
3. Install `Ubuntu` or `Ubuntu 24.04 LTS`.
4. Launch Ubuntu once and create the Linux user/password.

Restart Windows if WSL asks for it. Then open Ubuntu and check GPU visibility:

```bash
nvidia-smi
```

If `wsl --install` opens a shell like this:

```text
root@DESKTOP-...:~#
```

that is fine for this workflow. You can either continue as `root`, or later
create a normal Linux user. If Windows still says there is no default WSL
distribution, leave the Ubuntu shell with:

```bash
exit
```

Then run in Windows PowerShell:

```powershell
wsl -l -v
```

If Ubuntu is listed but not default, set it explicitly:

```powershell
wsl --set-default Ubuntu
```

If the listed name is different, use that exact name. You can also put this
exact name in SynCT's `WSL distro` field instead of leaving it blank.

The warning below is common when Windows has a localhost proxy:

```text
wsl: detected localhost proxy configuration but not mirrored into WSL
```

It does not prevent GPU use. It only means `apt` or `pip` may need proxy/network
configuration if downloads fail.

## 2. Create the SynthMorph GPU environment

In Ubuntu, go to this module folder through `/mnt/d/...` and run:

```bash
apt update
apt install -y python3-venv python3-pip
cd "/mnt/d/hkj/Slicer 5.8.1/Slicer 5.8.1/slicer.org/Extensions-33241/SynCT/SynCT/mri_synthmorph"
bash setup_wsl_gpu.sh
```

The setup uses the local `synthmorph` and `voxelmorph` folders in this module
and installs the remaining Python dependencies from PyPI.

The default Python path created by the script is:

```text
$HOME/envs/synthmorph-gpu/bin/python
```

If you ran the setup as `root`, the script may print:

```text
WSL SynthMorph GPU environment is ready: /root/envs/synthmorph-gpu/bin/python
```

In that case, use this explicit path in SynCT's `WSL Python` field:

```text
/root/envs/synthmorph-gpu/bin/python
```

## 3. Verify TensorFlow GPU

```bash
$HOME/envs/synthmorph-gpu/bin/python - <<'PY'
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
PY
```

The GPU list must not be empty.

From Windows PowerShell, use single quotes around the Linux command to avoid
nested quoting errors:

```powershell
wsl -d Ubuntu -- bash -lc 'export CUDA_VISIBLE_DEVICES=1; /root/envs/synthmorph-gpu/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices(chr(71)+chr(80)+chr(85)))"'
```

If your distribution name is different, replace `Ubuntu` with the exact name
shown by `wsl -l -v`.

Messages such as `Unable to register cuDNN factory` are TensorFlow startup
warnings and are not fatal if the GPU list is shown. If TensorFlow reports that
compute capability 12.0 kernels will be JIT-compiled, it is usually referring
to an RTX 50-series GPU. Prefer `CUDA_VISIBLE_DEVICES=1` to use the RTX 4090 on
this workstation and avoid the long first-run JIT compile on the RTX 5090.

Some neurite/voxelmorph releases still import Keras 2 names such as
`mean_absolute_error`. SynCT patches these aliases at SynthMorph startup so the
local WSL environment can run with TensorFlow/Keras 2.19.

## 4. Use from SynCT

In the SynCT module:

- Set `SynthMorph backend` to `WSL GPU`.
- Leave `WSL distro` blank to use the default WSL distro, or set it to the exact
  name shown by `wsl -l -v`, usually `Ubuntu`.
- Set `WSL Python` to `$HOME/envs/synthmorph-gpu/bin/python`, or to
  `/root/envs/synthmorph-gpu/bin/python` if the setup was run as `root`.
- Set `CUDA_VISIBLE_DEVICES` to `1` to prefer the RTX 4090 on this workstation, or `0` for the RTX 5090.

SynCT automatically converts Windows paths such as `D:\...` to WSL paths such as `/mnt/d/...`.
