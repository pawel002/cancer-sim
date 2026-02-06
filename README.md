# Modeling Cancer Cells Growth with Hypoxia

### Task Description

We can describe the evolution of cancer cells over certain domain using the following PDE described in
[reference]:

$$
    \frac{\partial u(x, t)}{\partial t} = D \nabla^2 u + \rho u(1 - \frac{u}{K}) - \beta R(x, t) H(x) u,
$$

where:

- $x$ - spatial dimension (can be linear, 2D or 3D, depeding on simulation).
- $u$ - tumor cell density function, normalized to $K=1$.
- $R(x, t)$ - radiation dose distribution.
- $H(x)$ - hypoxia function.
- $\beta$ - radiosensitivity coefficient (sensitivity to radiation).

Parts of the equation represent:

- $D \nabla^2$ - motility of cancer cells (ability to move from one location to another)
- $\rho u(1-\frac{u}{K})$ - poliferation of cells (the increase in cell number through cell growth and division).
- $\beta R(x,t) H(x) u$ - the rate of killing cancer cells, weighted by hypoxia.

### PDE Simulation

### How to run the project

Python environment is handled using [uv](https://github.com/astral-sh/uv). To activate it run:

```bash
uv sync
source .venv/bin/activate
```

This repository uses precommit hooks to keep the code clean.

---

### Bibliography

1. sadf
