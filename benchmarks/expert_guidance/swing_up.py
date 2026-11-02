"""The expert of the expert-guidance experiment (PREREG.md): an energy
swing-up handing over to a PD near the top, for gymnax's Pendulum-v1.

Pendulum-v1 is a uniform rod of mass m = 1 and length l = 1, driven at its
pivot by a torque u in [-2, 2], under g = 10; the angle is 0 upright and
the observation is [cos, sin, angular velocity]. Its energy, 0 at the
pivot's height,

    E = (m l^2 / 3) thdot^2 / 2 + m g (l / 2) cos(theta) = thdot^2 / 6 + 5 cos(theta),

is 5 upright at rest, and dE/dt = u thdot.

Away from the top, Astrom and Furuta's energy control (Automatica 36(2),
2000, Eq. 8) pumps E towards the upright's: u = sat(k (E0 - E) sign(.)).
Their input is the pivot's acceleration, which moves E through
thdot cos(theta); a torque moves it through thdot alone, so the sign here
is thdot's. Near the top, where the torque limit can still hold the pole
up (15 sin(theta) < 3 * 2, |theta| < 0.41 rad), a PD catches it, as the
paper advises (a catching controller near the upright).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

UPRIGHT_ENERGY = 5.0  # m g l / 2
MAX_TORQUE = 2.0


@dataclass(frozen=True)
class SwingUpPD:
    """The expert, stateless: ``expert(obs) -> torque``, in the env's units.

    Gains, set from the model before any measurement (PREREG.md):
    energy_gain 1 (the torque saturates on most of the swing anyway);
    catch_angle 0.4 rad, just inside the 0.41 rad the torque can hold;
    kp 10 and kd 2, which make the linearised pole thddot = 15 theta + 3 u
    into thddot = -15 theta - 6 thdot, natural frequency 3.9 rad/s and
    damping 0.77.
    """

    energy_gain: float = 1.0
    catch_angle: float = 0.4
    kp: float = 10.0
    kd: float = 2.0

    def __call__(self, obs: jax.Array) -> jax.Array:
        cos, sin, velocity = obs[..., 0:1], obs[..., 1:2], obs[..., 2:3]
        angle = jnp.arctan2(sin, cos)
        energy = velocity**2 / 6.0 + UPRIGHT_ENERGY * cos
        # At rest at the bottom thdot is 0 and so would be the push: give it one.
        direction = jnp.where(velocity >= 0.0, 1.0, -1.0)
        swing = self.energy_gain * (UPRIGHT_ENERGY - energy) * direction
        catch = -(self.kp * angle + self.kd * velocity)
        torque = jnp.where(jnp.abs(angle) < self.catch_angle, catch, swing)
        return jnp.clip(torque, -MAX_TORQUE, MAX_TORQUE)
