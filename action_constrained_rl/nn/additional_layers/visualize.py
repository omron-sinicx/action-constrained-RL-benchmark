import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from ...constraint import LinearConstraint

class TestConstraint(LinearConstraint):
    def __init__(self, err=1e-3):
        self.err = err

    def numConstraints(self):
        return 4

    def E(self, state):
        return self.C(state)

    def f(self, state):
        return self.d(state)

    def tensor_C(self, state):
        return torch.tensor([[1.,1.],[-2.,1.],[0.,-1.],[1.,0.]]).repeat(state.shape[0], 1, 1)
  
    def tensor_d(self, state):
        return torch.tensor([1.,1.,1.,1.]).repeat(state.shape[0], 1)
    

if __name__ == "__main__":

    from .alpha_projection import AlphaProjectionLayer
    from .radial_shrinkage import ShrinkageLayer
    from .radial_shrinkage import SmoothShrinkageLayer
    import torch
    import matplotlib.path

    
    cons = TestConstraint()
    a_layer = AlphaProjectionLayer(cons)
    s_layer = ShrinkageLayer(cons)
    ss_layer = SmoothShrinkageLayer(cons)
    N=100
    actions = torch.Tensor(N, 2)
    states = torch.Tensor(N, 2)
    for i in range(N):
        actions[i] = torch.Tensor(3 * torch.rand(2) - torch.tensor([1.5, 1.5]))
        states[i] = torch.Tensor([0.,0.])
    for layer in [ss_layer, s_layer, a_layer]:
        res = layer.forward(actions, states)
        fig, ax = plt.subplots()

        ax.add_patch(patches.Polygon([[0,1], [-1,-1], [1, -1], [1,0]], fill = False))
        plt.xlim(-2,2)
        plt.ylim(-2,2)

        for i in range(N):
            ver1 = actions[i].detach().numpy()
            ver2 = res[i].detach().numpy()
            path = matplotlib.path.Path([ver1, ver2])
            ax.add_patch(patches.Circle(ver1, radius = 0.01, color = 'r'))
            ax.add_patch(patches.Circle(ver2, radius = 0.01, color = 'b'))
            ax.add_patch(patches.PathPatch(path))
        plt.show()
