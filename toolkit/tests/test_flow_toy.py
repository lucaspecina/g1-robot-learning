import torch

from toolkit.flow import toy


def test_demos_respect_the_exact_proportion():
    demos = toy.sample_demos(400, left_fraction=0.7)
    assert demos.shape == (400, 2)
    assert int((demos[:, 0] < toy.TREE_CENTER[0]).sum()) == 280


def test_nobody_goes_through_the_tree():
    demos = toy.sample_demos(2000)
    distance = torch.linalg.vector_norm(demos - torch.tensor(toy.TREE_CENTER), dim=1)
    assert bool((distance > toy.TREE_RADIUS).all())
