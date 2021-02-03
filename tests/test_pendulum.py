#!/usr/bin/env python3

from source.pendulum import Pendulum

import numpy
import pytest


def test_pendulum():
    omega, theta = 0.1, numpy.pi / 4
    l = 2.2
    t = 1
    u = (theta, omega)

    exp_derivative = Pendulum(l)
    expected = (0.1, -3.15305341975)
    actual = exp_derivative(t, u)

    for exp, act in zip(expected, actual):
        assert(act == pytest.approx(exp))


def test_pendulum_at_rest():
    omega = theta = 0
    l = 2.2
    t = 1
    u = (theta, omega)

    exp = Pendulum(l)
    act = (0, 0)

    assert(act == pytest.approx(exp(t, u)))


def test_zero_arrays():
    u = (0, 0)
    T = 10
    dt = 0.01

    pendulum = Pendulum()
    pendulum.solve(u, 10, 0.01)

    expected = numpy.zeros(int(numpy.ceil(T / dt)) + 1)
    actual = pendulum.theta

    numpy.testing.assert_array_equal(expected, actual)


def test_pendulum_radius():
    omega, theta = 0.1, numpy.pi / 4
    l = 2.2
    u = (theta, omega)
    t = 1

    pendulum = Pendulum(l)
    pendulum(t, u)
    pendulum.solve([numpy.pi / 4.0, 0.5 * numpy.pi], 10, 0.01)

    actual = numpy.full(10, l**2)
    rsq = (pendulum.x**2 + pendulum.u**2)
    expected = rsq

    for exp, act in zip(expected, actual):
        assert(act == pytest.approx(exp))
