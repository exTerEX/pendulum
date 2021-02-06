#!/usr/bin/env python3

from source.pendulum import Pendulum

# pyright: reportMissingImports=false
import numpy


class DampenedPendulum(Pendulum):
    def __init__(self, mass: float = 1.0, length: float = 1.0,
                 g: float = 9.81, R: float = 0.1) -> None:
        """Base pendulum object for simulating Pendulum movement.

        Args:
            length (float, optional): Length of the rod. Defaults to 1.0.
            mass (float, optional): Mass of bob. Defaults to 1.0.
            g (float, optional): Gravity constant. Defaults to 9.81.
            R (float, optional): Resistance constant. Defaults to 0.1.
        """
        self._length = float(length)
        self._mass = float(mass)
        self._gravity_constant = float(g)
        self._R = float(R)

    def __call__(self, t: float, u: tuple) -> tuple:
        """Call function.

        Args:
            t (float): Time
            u (tuple): Position as (theta, omega).

        Returns:
            tuple: Return the calculated next position.
        """
        constant = self._gravity_constant

        theta, omega = u

        dth1 = omega
        dth2 = -(constant / self._length) * numpy.sin(theta) - (self._R / self._mass) * omega

        return (dth1, dth2)
