import torch

from toolkit.flow import matching, toy


def test_ideal_compass_points_to_the_mean_at_the_start():
    demos = toy.sample_demos(400)
    x = torch.randn(10, 2)
    assert torch.allclose(matching.ideal_velocity(x, 0.0, demos), demos.mean(0) - x, atol=1e-5)


def test_one_euler_step_with_the_ideal_compass_lands_on_the_mean():
    # Con un solo paso, flow matching es la regresión de L6.1: todo cae en el promedio.
    demos = toy.sample_demos(400)
    x0 = torch.randn(50, 2)
    x1 = x0 + matching.ideal_velocity(x0, 0.0, demos)
    assert torch.allclose(x1, demos.mean(0).expand_as(x1), atol=1e-5)
