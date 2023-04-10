# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura

class ConstantFunction:
    def __init__(self, x):
        self.x = x

    def __call__(self, t):
        return self.x
