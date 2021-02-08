#!/usr/bin/env python3

from source.double_pendulum import DoublePendulum

# pyright: reportMissingImports=false
import numpy
import pytest


def test_double_pendulum():
    length1, length2 = 2.0, 1.0
    mass1, mass2 = 1.0, 4.0

    omega1, omega2 = 0.1, 0.5
    theta1, theta2 = numpy.pi / 4.0, numpy.pi / 2.0

    t, u = 1.0, (theta1, omega1, theta2, omega2)

    expected = (0.1, -1.031602, 0.5, -8.531904)
    actual = DoublePendulum(mass1, length1, mass2, length2)(t, u)

    assert(expected == pytest.approx(actual))


def test_double_pendulum_at_rest():
    length1, length2 = 2.0, 1.0
    mass1, mass2 = 1.0, 4.0

    omega1 = omega2 = theta1 = theta2 = 0

    t, u = 1.0, (theta1, omega1, theta2, omega2)

    expected = (0.0, 0.0, 0.0, 0.0)
    actual = DoublePendulum(mass1, length1, mass2, length2)(t, u)

    assert(expected == pytest.approx(actual))


def test_potential_energy():
    length1, length2 = 3, 0.5
    mass1, mass2 = 1.2, 2

    omega1, omega2 = 1.0, 0.2 * numpy.pi
    theta1, theta2 = numpy.pi / 4.0, numpy.pi / 3.0

    u = (theta1, omega1, theta2, omega2)
    t = 10

    doublePendulum = DoublePendulum(mass1, length1, mass2, length2)
    doublePendulum.solve(u, t, 1 / 60)

    expected = 32.488512
    actual = doublePendulum.potential[0]

    assert(expected == pytest.approx(actual))


def test_kinetic_energy():
    length1, length2 = 3, 0.5
    mass1, mass2 = 1.2, 2

    omega1, omega2 = 1.0, 0.2 * numpy.pi
    theta1, theta2 = numpy.pi / 4.0, numpy.pi / 3.0

    u = (theta1, omega1, theta2, omega2)
    t = 10

    doublePendulum = DoublePendulum(mass1, length1, mass2, length2)
    doublePendulum.solve(u, t, 1 / 60)

    expected = 15.707572
    actual = doublePendulum.kinetic[0]

    assert(expected == pytest.approx(actual))


def test_total_energy():
    length1, length2 = 3, 0.5
    mass1, mass2 = 1.2, 2

    omega1, omega2 = 1.0, 0.2 * numpy.pi
    theta1, theta2 = numpy.pi / 4.0, numpy.pi / 3.0

    u = (theta1, omega1, theta2, omega2)
    t = 10

    doublePendulum = DoublePendulum(mass1, length1, mass2, length2)
    doublePendulum.solve(u, t, 1 / 60)

    expected = 48.196083
    actual = doublePendulum.potential[0] + doublePendulum.kinetic[0]

    assert(expected == pytest.approx(actual))
