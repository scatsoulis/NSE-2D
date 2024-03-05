# Incompressible Navier-Stokes solver

Simple finite-difference Navier-Stokes solver for incompressible flow.

Features:

- Implementation in Python 3.11
- Fluid solver based on the pressure projection method with 3rd order Runge-Kutta time integration
- Structured grid
- 2D

## Environment

Install the following:

```
pip install poetry
poetry shell
poetry install
```

## Example run

From project root:

```
cd src/
python3 runSolver.py
```

## Acknowledgments

We thank Prof. Jane Bae for her guidance and support during this project (Ae232a - Caltech, 2022).
