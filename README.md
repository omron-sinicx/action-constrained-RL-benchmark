# Action Constrained Deep RL
## Docker
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

## Running
### Experiments
To solve **R-L2** task with by **DPre** algorithm in our paper with random seed 1 and logging to `logs/R+L2-DPre-1`, run:
```
python3 -m train --log_dir logs/R-L2-DPre-1 --prob_id R-L2 --algo_id DPre --seed 1
```
Note that you can also explicitly indicate tasks, algorithms or hyperparameters by arguments.
### Evaluation
When experiments with 1-10 seeds are logged in `logs/R-L2-DPre-1`, ..., `logs/R-L2-DPre-10`, run:
```
python3 -m evaluation --log_dir logs/R-L2-DPre --prob_id R-L2 --algo_id DPre
```
Then the evaluarion results are stored in `logs/R-L2-DPre`.