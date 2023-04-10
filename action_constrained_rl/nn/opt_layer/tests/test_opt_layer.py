import torch
from ..opt_layer import OptLayer
from pytest import approx
from ....half_cheetah.half_cheetah_dynamic_constraint import HalfCheetahDynamicConstraint

def test_opt_layer_half_cheetah():
    import numpy as np
    cons = HalfCheetahDynamicConstraint()
    layer = OptLayer(cons)
    a = torch.tensor(5.0 * np.ones(6))
    s = torch.tensor(np.ones(17))
    assert layer(a, s).sum() == approx(20.0)