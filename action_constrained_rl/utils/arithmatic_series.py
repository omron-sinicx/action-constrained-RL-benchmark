# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

class ArithmaticSeries:
    def __init__(self, a0, d):
        self.a0 = a0
        self.d = d

    def __call__(self, t):
        assert(t > 0)
        return self.a0 + (t-1) * self.d

if __name__ == "__main__":
    a0 = 0.001
    d = 0.02
    a = ArithmaticSeries(a0, d)
    print(a(1))
    print(a(2))
    print(a(100))
