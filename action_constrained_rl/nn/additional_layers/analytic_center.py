import cvxpy
import torch
def calc_analytic_center(C, d):
    n = C.shape[1]
    x = cvxpy.Variable(n)
    obj = cvxpy.Maximize(cvxpy.sum(cvxpy.log((d - C @ x))))
    prob = cvxpy.Problem(obj, [C @ x <= d])
    prob.solve()
    return x.value
