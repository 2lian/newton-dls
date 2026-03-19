# newton-dls
Transitioning to isaac. Currently broken because I pushed in a hurry

## Deps

Dowload the dependencies inside this repo

```bash
mkdir -p deps
git clone https://github.com/newton-physics/newton.git deps/newton
# BUG: you need to comment all entries with `open3d` in the newton pyproject.toml
git clone -b develop --depth 1 git@github.com:isaac-sim/IsaacLab.git deps/IsaacLab 
```

## Installation

Installation is [done through Pixi](https://pixi.prefix.dev/latest/)

```bash
pixi install
pixi run python -c "import isaacsim" # just to accept the EULA otherwise it is very very bugged
```

## Run

### Start rerun

```bash
pixi run isaaclab --sim
```
