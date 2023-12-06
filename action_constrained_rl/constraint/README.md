This folder contains scripts for constraints.

`Constraint` class in `constraint.py` is an abstract class for all constraints.

`LinearConstraint` class in `constraint.py` is an abstract class for constraints with the form $Cx\leq d$ for action $x$, where $C$ is a matrix and $d$ is a vector.
$C$ and $d$ can be denendent on states.
To implement a concrete class for a linear constraint, you must implement the follwoing two methods. Let $D_S$ be the dimension of the state space and $D_A$ be the dimension of the action space.
- `tensor_C`: Input a tensor with shape $\textit{BatchSize}\times D_S$, which represents a batch of states, and return a tensor with shape $\textit{BatchSize}\times \textit{NumberOfInequlity} \times D_A$, which represents corresponding $C$ matrices.
- `tensor_d`: Input a tensor with shape $\textit{BatchSize}\times D_S$, which represents a batch of states, and return a tensor with shape $\textit{BatchSize}\times \textit{NumberOfInequlity}$, which represents corresponding $d$ vectors.

`QuadraticConstraint` class in `quadratic_constraint.py` is an abstract class for constraints with the form $x^{T}Qx\leq M$ for action $x$, where $Q$ is a matrix and $M$ is a scalar.
$Q$ can be denendent on states.
To implement a concrete class for a quadratic constraint, you must implement the following method.
- `tensor_Q`: Input a tensor with shape $\textit{BatchSize}\times D_S$, which represents a batch of states, and return a tensor with shape $\textit{BatchSize}\times D_A \times D_A$, which represents corresponding $Q$ matrices.
