#!/usr/bin/env python3

from source.pendulum import Pendulum

# pyright: reportMissingImports=false
import numpy
import pytest


def test_pendulum_length():
    """Test for different length input values."""
    omega = 0.1
    theta = numpy.pi / 4.0

    length = (3.0, 1.8, 0.4)
    mass = 10.0
    t = 10.0

    expected = ((0.1, -2.312239), (0.1, -3.853732), (0.1, -17.341794))

    for index in range(len(expected)):
        actual = Pendulum(length[index], mass)(t, (theta, omega))

        assert(expected[index] == pytest.approx(actual))


def test_pendulum_mass():
    """Test for different mass input values."""
    omega = 0.1
    theta = numpy.pi / 4.0

    length = 2.0
    mass = (1.0, 10.0, 100.0)
    t = 10.0

    expected = ((0.1, -3.468359), (0.1, -3.468359), (0.1, -3.468359))

    for index in range(len(expected)):
        actual = Pendulum(length, mass[index])(t, (theta, omega))

        assert(expected[index] == pytest.approx(actual))


def test_pendulum_time():
    """Test for different time input values."""
    omega = 0.1
    theta = numpy.pi / 4.0

    length = 2.0
    mass = 10.0
    t = (8.0, 28.5, 58.0)

    expected = ((0.1, -3.468359), (0.1, -3.468359), (0.1, -3.468359))

    for index in range(len(expected)):
        actual = Pendulum(length, mass)(t[index], (theta, omega))

        assert(expected[index] == pytest.approx(actual))


def test_pendulum_theta():
    """Test for different theta input values."""
    omega = 0.1
    theta = (numpy.pi / 2.0, numpy.pi / 3.0, numpy.pi / 4.0)

    length = 2.0
    mass = 10.0
    t = 10.0

    expected = ((0.1, -4.905000), (0.1, -4.247855), (0.1, -3.468359))

    for index in range(len(expected)):
        actual = Pendulum(length, mass)(t, (theta[index], omega))

        assert(expected[index] == pytest.approx(actual))


def test_pendulum_omega():
    """Test for different omega input values."""
    omega = (0.1, 0.5, 0.9)
    theta = numpy.pi / 4.0

    length = 2.0
    mass = 10.0
    t = 10.0

    expected = ((0.1, -3.468359), (0.5, -3.468359), (0.9, -3.468359))

    for index in range(len(expected)):
        actual = Pendulum(length, mass)(t, (theta, omega[index]))

        assert(expected[index] == pytest.approx(actual))


def test_pendulum_radius():
    """Test that pendulum radius is correct in all timepoints."""
    omega, theta = 0.1, numpy.pi / 4

    length, mass, t, dt = 2.0, 10.0, 10, 0.1

    expected = [4.0] * 101

    pendulum = Pendulum(length, mass)
    pendulum(t, (theta, omega))
    pendulum.solve([numpy.pi / 4.0, 0.5 * numpy.pi], t, dt)

    actual = (pendulum.x**2 + pendulum.u**2)

    assert(expected == pytest.approx(actual))


def test_pendulum_at_rest():
    """Test to ensure that pendulum stay at rest."""
    omega = theta = 0

    length, mass, t = 2.0, 8.0, 20.0

    expected = (0.0, 0.0)

    pendulum = Pendulum(length, mass)
    actual = pendulum(t, (theta, omega))

    assert(expected == pytest.approx(actual))
