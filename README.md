# newton-dls
Transitioning to isaac?

## Deps

Dowload the dependencies inside this repo

```bash
mkdir -p deps
git clone https://github.com/newton-physics/newton.git deps/newton
# BUG: you need to comment all entries with `open3d` in the newton pyproject.toml
```

## Installation

Installation is [done through Pixi](https://pixi.prefix.dev/latest/)

```bash
pixi install
```

you might need to change the cuda version in the pixi.toml

## Run

### Start rerun

```bash
pixi run rerun
```

### Start the sim

```bash
pixi run python -m newton_dls.debug
```
