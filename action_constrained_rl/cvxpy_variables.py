# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura
class CVXPYVariables:
    """
    class to store cvxpy proplem data
    """
    def __init__(self, x, q, s, cons, obj, prob):
        self.x = x
        self.q = q
        self.s = s
        self.cons = cons
        self.obj = obj
        self.prob = prob
