"""
Tests for functionality checks in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import pytest
import numpy as np


@pytest.fixture
def solver():
    solver = SolveDiffusion2D()
    return solver


def test_initialize_physical_parameters(solver):
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    input_w = 100.
    input_h = 200.
    input_dx = 0.1
    input_dy = 0.1

    expected_T_cold = 100.
    expected_T_hot = 1000.
    expected_D = 5.

    expected_dt = 5 * 10e-5

    solver.initialize_domain(input_w, input_h, input_dx, input_dy)
    solver.initialize_physical_parameters(expected_D, expected_T_cold, expected_T_hot)

    assert solver.dt == pytest.approx(expected_dt, rel=1e-12, abs=0.0)


def test_set_initial_condition(solver):
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    expected_nx = 1000
    expected_ny = 2000

    input_w = 100.
    input_h = 200.
    input_dx = 0.1
    input_dy = 0.1

    expected_T_cold = 100.
    expected_T_hot = 1000.
    expected_D = 5.
    
    solver.initialize_domain(input_w, input_h, input_dx, input_dy)
    solver.initialize_physical_parameters(expected_D, expected_T_cold, expected_T_hot)
    calculated_u = solver.set_initial_condition()

    expected_u = solver.T_cold * np.ones((expected_nx, expected_ny))

    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(expected_nx):
        for j in range(expected_ny):
            p2 = (i * input_dx - cx) ** 2 + (j * input_dy - cy) ** 2
            if p2 < r2:
                expected_u[i, j] = solver.T_hot

    assert np.array_equal(expected_u, calculated_u)
    