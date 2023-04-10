# Action Constrained Deep RL
This repository contains various environments for aciton-constrained RL problems, as well as the implementation of various algorithms for action-constrained deep reinforcement learning. The code is organized for ease of use and experimentation.

## Installation
We recommend docker-based installation.

### Building Image
```
docker build -t action_constrained_rl --build-arg USERNAME=$(USERNAME) --build-arg USER_UID=$(USER_UID) .
```

or 

```
make docker_build
```

### Attaching to a Container

```
docker run --gpus all -it -v $(pwd):/workspace/action_constrained_rl action_constrained_rl:latest
```

## Available Environment and Constraints
We list the available environment and constraint combinations in the respository:
| Environment  | Name   | Constraint                                         |
|--------------|--------|----------------------------------------------------|
| Reacher      | R+N    | No additional constraint                           |
|              | R+L2   | $$a_1^2+a_2^2\leq 0.05$$                              |
|              | R+O03  | $$\sum_{i=1}^2 \|w_ia_i\|\leq 0.3$$                   |
|              | R+O10  | $$\sum_{i=1}^2 \|w_ia_i\|\leq 1.0$$                   |
|              | R+O30  | $$\sum_{i=1}^2 \|w_ia_i\|\leq 3.0$$                   |
|              | R+M    | $$\sum_{i=1}^2 \max\{w_ia_i,0\}\leq 1.0$$             |
|              | R+T    | $$a_1^2+2a_1(a_1+a_2)\cos \theta_2+(a_1+a_2)^2\leq 0.05$$|
| HalfCheetah  | HC+O   | $$\sum_{i=1}^6\|w_ia_i\|\leq 20$$                     |
|              | HC+MA  | $w_1a_1\sin (\theta_1+\theta_2+\theta_3)$          |
|              |        | $+w_4a_4\sin (\theta_4+\theta_5+\theta_6)\leq 5$    |
| Hopper       | H+M    | $\sum_{i=1}^3\max\{w_ia_i,0\}\leq 10$               |
|              | H+O+S  | $\sum_{i=1}^3\|w_ia_i\|\leq 10$, $\sum_{i=1}^3 a_i^2\sin^2\theta_i\leq 0.1$ |
| Walker2d     | W+M    | $\sum_{i=1}^6\max\{w_ia_i,0\}\leq 10$               |
|              | W+O+S  | $\sum_{i=1}^6\|w_ia_i\|\leq 10$, $\sum_{i=1}^6 a_i^2\sin^2\theta_i\leq 0.1$ |


## Example
### Running
To run the **DPre** algorithm on the **R-L2** task with a random seed of 1 and log the results to `logs/R+L2-DPre-1`, execute the following command:
```
python3 -m train --log_dir logs/R-L2-DPre-1 --prob_id R-L2 --algo_id DPre --seed 1
```
Note that you can also explicitly specify tasks, algorithms, or hyperparameters using command-line arguments.

### Aggregating Results
When experiments with 1-10 seeds are logged in `logs/R-L2-DPre-1`, ..., `logs/R-L2-DPre-10`, run:
```
python3 -m evaluation --log_dir logs/R-L2-DPre --prob_id R-L2 --algo_id DPre
```
Then the evaluarion results are stored in `logs/R-L2-DPre`.