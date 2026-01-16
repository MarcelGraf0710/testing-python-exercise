"""
Tests for functions in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import pytest
import numpy as np


@pytest.fixture
def solver():
    solver = SolveDiffusion2D()
    return solver

def test_initialize_domain(solver: SolveDiffusion2D):
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    expected_nx = 50
    expected_ny = 100

    input_w = 100.
    input_h = 200.
    input_dx = 2.
    input_dy = 2.
    
    solver.initialize_domain(input_w, input_h, input_dx, input_dy)
    assert solver.nx == expected_nx 
    assert solver.ny == expected_ny
    

def test_initialize_physical_parameters(solver: SolveDiffusion2D):
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    expected_T_cold = 100.
    expected_T_hot = 1000.
    expected_D = 5.

    expected_dt = 5 * 10e-5

    solver.dx = 0.1
    solver.dy = 0.1

    solver.initialize_physical_parameters(expected_D, expected_T_cold, expected_T_hot)

    assert solver.T_cold == expected_T_cold 
    assert solver.T_hot == expected_T_hot
    assert solver.D == expected_D
    assert solver.dt == pytest.approx(expected_dt, rel=1e-12, abs=0.0)


def test_set_initial_condition(solver: SolveDiffusion2D):
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver.nx = 200
    solver.ny = 200
    solver.dx = 0.1
    solver.dy = 0.1
    solver.T_cold = 50
    solver.T_hot = 200

    expected_u = solver.T_cold * np.ones((solver.nx, solver.ny))

    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(solver.nx):
        for j in range(solver.ny):
            p2 = (i * solver.dx - cx) ** 2 + (j * solver.dy - cy) ** 2
            if p2 < r2:
                expected_u[i, j] = solver.T_hot

    calculated_u = solver.set_initial_condition()
    assert np.array_equal(expected_u, calculated_u)
