# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

import numpy as np
class LogSeries:
    def __init__(self, a0):
        self.a0 = a0

    def __call__(self, t):
        assert(t > 0)
        return self.a0 + np.log(t)

if __name__ == "__main__":
    a0 = 0.1
    a = LogSeries(a0)
    print(a(1))
    print(a(2))
    print(a(3))
    print(a(100))
    print(a(1000))
