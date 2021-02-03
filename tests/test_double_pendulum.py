#!/usr/bin/env python3

from source.double_pendulum import DoublePendulum

import numpy
import pytest


def test_double_pendulum():
    length1, length2 = 2, 1
    mass1, mass2 = 1, 4

    omega1, omega2 = 0.1, 0.5
    theta1, theta2 = numpy.pi / 4, numpy.pi / 2

    u = (theta1, omega1, theta2, omega2)
    t = 1

    exp_derivative = DoublePendulum(mass1, length1, mass2, length2)

    expected = exp_derivative(t, u)
    actual = (0.1, -1.03160179038, 0.5, -8.53190355937)

    for exp, act in zip(expected, actual):
        assert(act == pytest.approx(exp))


def test_double_pendulum_at_rest():
    length1, length2 = 2, 1
    mass1, mass2 = 1, 4

    omega1 = omega2 = theta1 = theta2 = 0

    u = (theta1, omega1, theta2, omega2)
    t = 1

    expected = DoublePendulum(mass1, length1, mass2, length2)
    actual = (0, 0, 0, 0)

    assert(actual == pytest.approx(expected(t, u)))


def test_potential_energy():
    length1, length2 = 3, 0.5
    mass1, mass2 = 1.2, 2

    omega1, omega2 = 1.0, 0.2 * numpy.pi
    theta1, theta2 = numpy.pi / 4.0, numpy.pi / 3.0

    u = (theta1, omega1, theta2, omega2)
    t = 10

    doublePendulum = DoublePendulum(mass1, length1, mass2, length2)
    doublePendulum.solve(u, t, 1 / 60)

    expected = doublePendulum.potential[0]
    actual = 33.531572

    assert(actual == pytest.approx(expected, abs=2.5e-2))


def test_kinetic_energy():
    length1, length2 = 3, 0.5
    mass1, mass2 = 1.2, 2

    omega1, omega2 = 1.0, 0.2 * numpy.pi
    theta1, theta2 = numpy.pi / 4.0, numpy.pi / 3.0

    u = (theta1, omega1, theta2, omega2)
    t = 10

    doublePendulum = DoublePendulum(mass1, length1, mass2, length2)
    doublePendulum.solve(u, t, 1 / 60)

    expected = doublePendulum.kinetic[0]
    actual = 14.141519

    assert(actual == pytest.approx(expected, abs=2.5e-2))


def test_total_energy():
    l1 = 3
    m1 = 1.2
    l2 = 0.5
    m2 = 2

    omega1 = 0.5
    theta1 = numpy.pi / 4
    omega2 = 0.2
    theta2 = numpy.pi / 3

    u = (theta1, omega1, theta2, omega2)
    t = 10

    doublePendulum = DoublePendulum(m1, l1, m2, l2)
    doublePendulum.solve(u, t, 1 / 60)

    expected = doublePendulum.potential[0] + doublePendulum.kinetic[0]
    actual = 50.393061

    assert(actual == pytest.approx(expected, abs=2.5e-2))
