#!/usr/bin/env python3

from source.pendulum import Pendulum

import numpy
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

default_kwargs = {
    "gravity_constant": 9.81
}


class DoublePendulum:
    def __init__(self, mass1: float = 1.0, length1: float = 1.0,
                 mass2: float = 1.0, length2: float = 1.0, **kwargs):
        kwargs = {**default_kwargs, **kwargs}

        self._mass1 = mass1
        self._mass2 = mass2
        self._length1 = length1
        self._length2 = length2

        self._gravity_constant = kwargs["gravity_constant"]

    def __call__(self, t: float, u: tuple) -> tuple:
        m1, l1, m2, l2 = self._mass1, self._length1, self._mass2, self._length2
        g = self._gravity_constant

        (theta1, omega1, theta2, omega2) = u

        self._theta1 = theta1
        self._theta2 = theta2

        self._omega1 = omega1
        self._omega2 = omega2

        dht1, dht2 = omega1, omega2

        dho1 = ((m2 * l1 * omega1 ** 2 * numpy.sin(theta2 - theta1)) * numpy.cos(theta2 - theta1) +
                (m2 * g * numpy.sin(theta2) * numpy.cos(theta2 - theta1)) +
                (m2 * l2 * omega2 ** 2 * numpy.sin(theta2 - theta1)) -
                ((m1 + m2) * g * numpy.sin(theta1))) / (((m1 + m2) * l1) -
                                                        (m2 * l1 * numpy.cos(theta2 - theta1) ** 2))

        dho2 = (-(m2 * l1 * omega2 ** 2 * numpy.sin(theta2 - theta1) * numpy.cos(theta2 - theta1)) +
                ((m1 + m2) * g * numpy.sin(theta1) * numpy.cos(theta2 - theta1)) -
                ((m1 + m2) * l1 * omega1 ** 2 * numpy.sin(theta2 - theta1)) -
                ((m1 + m2) * g * numpy.sin(theta2))) / (((m1 + m2) * l2) -
                                                        (m2 * l2 * numpy.cos(theta2 - theta1) ** 2))

        return (dht1, dho1, dht2, dho2)

    def solve(self, u0, T, dt, angular_unit: str = "rad"):
        if angular_unit == "deg":
            u0 *= numpy.pi / 180

        t = numpy.linspace(0, T, int(numpy.ceil(T / dt)) + 1, dtype=float)
        solution = scipy.integrate.solve_ivp(self, [0, T], u0, method="Radau", t_eval=t)

        self._dt = dt
        self._t, self._theta1, self._theta2 = solution.t, solution.y[0], solution.y[2]

    @property
    def t(self):
        return self._t

    @property
    def theta1(self):
        return self._theta1

    @property
    def omega1(self):
        return self._omega1

    @property
    def theta2(self):
        return self._theta2

    @property
    def theta2(self):
        return self._omega2

    @property
    def x1(self):
        return self._length1 * numpy.sin(self.theta1)

    @property
    def u1(self):
        return -self._length1 * numpy.cos(self.theta1)

    @property
    def x2(self):
        return self.x1 + self._length2 * numpy.sin(self.theta2)

    @property
    def u2(self):
        return self.u1 - self._length2 * numpy.cos(self.theta2)

    @property
    def vx1(self):
        return numpy.gradient(self.x1, self.t)

    @property
    def vu1(self):
        return numpy.gradient(self.u1, self.t)

    @property
    def vx2(self):
        return numpy.gradient(self.x2, self.t)

    @property
    def vu2(self):
        return numpy.gradient(self.u2, self.t)

    @property
    def potential(self):
        p1 = self._mass1 * self._gravity_constant * (self.u1 + self._length1)
        p2 = self._mass2 * self._gravity_constant * (self.u2 + self._length1 + self._length2)

        return p1 + p2

    @property
    def kinetic(self):
        k1 = 0.5 * self._mass1 * (self.vx1**2 + self.vu1**2)
        k2 = 0.5 * self._mass2 * (self.vx2**2 + self.vu2**2)

        return k1 + k2
