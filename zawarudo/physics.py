import jax.numpy as jnp


class Constants:
    grav = 1e0
    spring = 2e3
    drag = 1e0
    mass = 2e-1
    radius = 1e-1


def cap_vector(x, max_norm):
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return jnp.where(norm > max_norm, x / norm * max_norm, x)


def du(t, u):
    # (n, 3)
    pos = u[:, 0, :]

    # (n, 3)
    vel = u[:, 1, :]

    # (n, n, 3)
    b2b = pos[:, jnp.newaxis, :] - pos[jnp.newaxis, :, :]

    # (n, n, 3)
    b2b_norm = jnp.nan_to_num(b2b / jnp.linalg.norm(b2b, axis=-1, keepdims=True))

    # (n, n, 3)
    b2b_vel = vel[:, jnp.newaxis, :] - vel[jnp.newaxis, :, :]

    # (n, n, 3)
    dist2 = jnp.sum(b2b ** 2, axis=-1, keepdims=True)

    # (n, n, 3)
    dist = jnp.sqrt(dist2)

    # (n, n, 3)
    grav = jnp.nan_to_num(Constants.grav * Constants.mass ** 2 / dist2) * b2b_norm

    # (n, n, 3)
    spring = - (Constants.spring * (2 * Constants.radius - dist)) * b2b_norm

    # (n, n, 3)
    drag = Constants.drag * b2b_vel

    # (n, n, 3)
    colliding = dist < 2 * Constants.radius

    # (n, n, 3)
    force = jnp.where(colliding, spring + drag, 0) + grav

    # (n, 3)
    acc = jnp.sum(force, axis=0) / Constants.mass

    du = jnp.stack((
        vel, acc,
        # cap_vector(vel, 1e4),
        # cap_vector(acc, 1e4),
    ), axis=1)

    return du
