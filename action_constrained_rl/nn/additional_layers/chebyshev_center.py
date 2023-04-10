# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

import gurobipy as gp
import numpy as np

def calc_chebyshev_center(C, d):
    m, n = C.shape
    with gp.Model() as model:
        x = []
        for i in range(n):
            x.append(model.addVar(lb = -1, ub = 1, vtype = gp.GRB.CONTINUOUS))
        r = model.addVar(lb = 0, ub = gp.GRB.INFINITY, vtype = gp.GRB.CONTINUOUS)
        model.setObjective(r, sense = gp.GRB.MAXIMIZE)
        norms = np.linalg.norm(C, axis=1)
        for j in range(m):
            exp = gp.LinExpr()
            for i in range(n):
                exp+=C[j,i]*x[i]
            exp+=norms[j]*r
            model.addConstr(exp <= d[j])
        model.optimize()
        x_value = np.array(model.X[0:n])
    return x_value
