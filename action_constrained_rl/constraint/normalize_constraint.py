# Copyright (c) 2023 OMRON SINIC X Corporation
# Author: Shuwa Miura, Kazumi Kasaura
import numpy as np
def normalizeConstraint(A, b):
    norms = (np.linalg.norm(A, axis=1))
    return (A/norms[:,None], b/norms)


if __name__ == "__main__":
    import numpy as np
    A = np.array([[3.0, 4.0], [50.0, 3.0]])
    b = np.array([1.0, 2.0])
    norms = (np.linalg.norm(A, axis=1))
    print(A / norms)
    print(b / norms)
