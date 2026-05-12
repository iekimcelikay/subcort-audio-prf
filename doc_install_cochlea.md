
## Installing `cochlea` (zilany2014 model) on dIPC Servers

### 1. Clone the `py3-zilany2014` branch into your project

```bash
git clone --branch py3-zilany2014 https://github.com/iekimcelikay/cochlea.git /scratch/<username>/workspace/subcort-audio-prf/zilany2014
```

### 2. Add a `pyproject.toml` to the cloned directory

The package uses Cython extensions. Modern `pip` creates isolated build environments that don't have access to your conda environment's packages, so we need to explicitly declare the build dependencies.

```bash
cd /scratch/<username>/workspace/subcort-audio-prf/zilany2014
```

Create a `pyproject.toml` file with the following content:

```toml
[build-system]
requires = ["setuptools", "Cython", "numpy"]
build-backend = "setuptools.build_meta"
```

### 3. Install in editable mode

Make sure your conda environment is activated, then run:

```bash
pip install -e . --no-build-isolation
```

The `--no-build-isolation` flag is needed to prevent `pip` from spawning an isolated subprocess that doesn't have access to your environment's Cython installation.

### Notes
- Tested with Python 3.9, Cython 3.2.4, setuptools 80.9.0
- A C compiler (`gcc`) must be available on the system (it is on dIPC servers)
- The editable install means any changes to the source code in `zilany2014/` are immediately reflected without reinstalling