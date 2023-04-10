# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

class GeometricSeries:
    def __init__(self, a0, gamma):
        self.a0 = a0
        self.gamma = gamma

    def __call__(self, t):
        assert(t > 0)
        return self.a0 * self.gamma ** (t-1)

if __name__ == "__main__":
    a0 = 1.0
    d = 2.0
    a = GeometricSeries(a0, d)
    print(a(1))
    print(a(2))
    print(a(10))
