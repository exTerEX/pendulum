#!/usr/bin/env python3

import numpy
import scipy.integrate


class Pendulum:
    def __init__(self, length: float = 1.0, mass: float = 1.0, g: float = 9.81) -> None:
        """Base pendulum object for simulating Pendulum movement.

        Args:
            length (float, optional): Length of the rod. Defaults to 1.0.
            mass (float, optional): Mass of bob. Defaults to 1.0.
            g (float, optional): Gravity constant. Defaults to 9.81.
        """
        self._length = float(length)
        self._mass = float(mass)
        self._gravity_constant = float(g)
        self._t, self._u = None, None

    def __call__(self, t: float, u: tuple) -> tuple:
        """Call function.

        Args:
            t (float): Time
            u (tuple): Position as (theta, omega).

        Returns:
            tuple: Return the calculated next position.
        """
        length, constant = self._length, self._gravity_constant

        (theta, omega) = u

        return (omega, -(constant / length) * numpy.sin(theta))

    def solve(self, u0: float, T: float, dt: float, angular_unit: str = "rad"):
        if angular_unit == "deg":
            u0 *= numpy.pi / 180

        t = numpy.linspace(0, T, int(numpy.ceil(T / dt)) + 1, dtype=float)

        solution = scipy.integrate.solve_ivp(self, [0, T], u0, method="Radau", t_eval=t)

        self._t, self._u = solution.t, solution.y

    @property
    def t(self):
        if self._t is None:
            raise ValueError("Solve method must be called.")

        return self._t

    @property
    def theta(self):
        if self._u is None:
            raise ValueError("Solve method must be called.")

        return self._u[0]

    @property
    def omega(self):
        if self._u is None:
            raise ValueError("Solve method must be called.")

        return self._u[1]

    @property
    def x(self):
        return self._length * numpy.sin(self._u[0])

    @property
    def u(self):
        return -self._length * numpy.cos(self._u[0])

    @property
    def vx(self):
        return numpy.gradient(self.x, self.t)

    @property
    def vu(self):
        return numpy.gradient(self.u, self.t)

    @property
    def potential(self):
        return self._mass * self._gravity_constant * (self.u + self._l)

    @property
    def kinetic(self):
        return 0.5 * self._mass * (self.vx**2 + self.vu**2)
