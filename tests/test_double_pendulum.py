#!/usr/bin/env python3

from source.double_pendulum import DoublePendulum

# pyright: reportMissingImports=false
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
    actual = 32.488512

    assert(actual == pytest.approx(expected))


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
    actual = 15.707572

    assert(actual == pytest.approx(expected))


def test_total_energy():
    l1 = 3
    m1 = 1.2
    l2 = 0.5
    m2 = 2

    omega1 = 1.0
    theta1 = numpy.pi / 4.0
    omega2 = numpy.pi / 5.0
    theta2 = numpy.pi / 3.0

    u = (theta1, omega1, theta2, omega2)
    t = 10

    doublePendulum = DoublePendulum(m1, l1, m2, l2)
    doublePendulum.solve(u, t, 1 / 60)

    expected = doublePendulum.potential[0] + doublePendulum.kinetic[0]
    actual = 48.196083

    assert(actual == pytest.approx(expected))
