import jax
import jax.numpy as jnp

from .physics import du


def rk4(t, u, dt):
    k1 = dt * du(t, u)
    k2 = dt * du(t + dt/2, u + k1/2)
    k3 = dt * du(t + dt/2, u + k2/2)
    k4 = dt * du(t + dt, u + k3)
    return u + (k1 + 2*k2 + 2*k3 + k4) / 6


def normalize(x):
    # return x.at[:, 0, :].set(x[:, 0, :] - jnp.mean(x[:, 0, :], axis=0, keepdims=True))
    # return x / jnp.linalg.norm(x, axis=1, keepdims=True)
    return x


def step(t, u, dt, n):
    for _ in range(n):
        u = normalize(rk4(t, u, dt))
        t = t + dt
    return u, t
