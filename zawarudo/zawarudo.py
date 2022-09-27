import jax
import jax.numpy as jnp

from . import physics
from . import solver


def whoop_planet(x0, y0, vx, vy, r):
    bodies = []
    for x in jnp.linspace(x0 - r, x0 + r, int(r / physics.Constants.radius)):
        for y in jnp.linspace(y0 - r, y0 + r, int(r / physics.Constants.radius)):
            if (x0 - x)**2 + (y0 - y)**2 < r**2:
                bodies.append([x, y, vx, vy])
    return jnp.reshape(jnp.array(bodies), (-1, 2, 2))


class ZaWarudo:
    def __init__(self):
        self._state = jnp.concatenate((
            whoop_planet(0, 0, 0, 0, 3),
            whoop_planet(100, 2, -20, 0, 1),
            whoop_planet(-100, 2, 20, 0, 1),
        ))

        print(f'Created {self._state.shape[0]} bodies')

        key = jax.random.PRNGKey(42)
        self._colors = jax.random.uniform(key, (self._state.shape[0], 3))
        self._colors /= jnp.linalg.norm(self._colors, axis=-1, keepdims=True)

        self._t = 0.0

    @property
    def planets(self):
        return zip(self._state[:, 0, :], self._colors)

    def step(self, dt: float):
        self._state = solver.step(self._t, self._state, dt)
        self._t += dt
