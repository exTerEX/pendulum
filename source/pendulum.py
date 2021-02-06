#!/usr/bin/env python3

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import numpy
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

default_kwargs = {
    "gravity_constant": 9.81
}


class Pendulum:
    def __init__(self, length: float = 1.0, mass: float = 1.0, **kwargs) -> None:
        """Base pendulum object for simulating Pendulum movement.

        Args:
            length (float, optional): Length of the rod. Defaults to 1.0.
            mass (float, optional): Mass of bob. Defaults to 1.0.
            gravity_constant (float, optional): Gravity constant. Defaults to 9.81.
        """
        kwargs = {**default_kwargs, **kwargs}

        self._length = float(length)
        self._mass = float(mass)
        self._gravity_constant = float(kwargs["gravity_constant"])
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

        self._dt = dt
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
        return self._mass * self._gravity_constant * (self.u + self._length)

    @property
    def kinetic(self):
        return 0.5 * self._mass * (self.vx**2 + self.vu**2)

    def animate(self):
        fig = plt.figure(figsize=(2, 2), dpi=200)

        plt.axis("equal")

        length = round(self._length * 1.15, 2)

        plt.ylim(-length, length)
        plt.xlim(-length, length)

        plt.grid(False)
        plt.tick_params(labelsize=8)

        self._pendulums, = plt.plot([], [], "o-", lw=2)

        self._Animation = animation.FuncAnimation(
            fig, self._next_frame, frames=range(len(self.x)),
            repeat=None, interval=100 * self._dt, blit=True)

        plt.close(fig)

    def _next_frame(self, index):
        self._pendulums.set_data(
            (0, self.x[index]),
            (0, self.u[index])
        )

        return self._pendulums,

    @property
    def get_animation(self):
        return self._Animation
