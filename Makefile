GIT_TREE_HASH := $(shell git write-tree)
USERNAME=$(shell whoami)
USER_UID=$(shell id -u $(USERNAME))
JOB_SCHEDULER := tsp -n

LOG_DIR := log
IMG_DIR := img
MAIN_SCRIPT := train.py
VERBOSITY := 0
ENVS := HalfCheetahDynamic Hopper-v2 IdealizedPendulum-v0 Reacher-v2
A0 := 0.001 0.005
R := 1.001 1.01 1.05
N := 10 100
LR := 0.01
#REPS := 0
REPS := 0 1 2 3 4
HalfCheetahDynamic_C := 0.0 0.001 0.005 0.01 0.025 0.05 0.075 0.1 0.5
HalfCheetahDynamic_NumTimeSteps := 1000000
HalfCheetahDynamic_EvalFreq := 5000
Hopper-v2_C := 0.0 0.005 0.01 0.025 0.05 0.075 0.1 0.5
Hopper-v2_NumTimeSteps := 100000
Hopper-v2_EvalFreq := 1000
IdealizedPendulum-v0_C := 0.0 0.005 0.01 0.025 0.05 0.075 0.1 0.5
IdealizedPendulum-v0_NumTimeSteps := 10000
IdealizedPendulum-v0_EvalFreq := 200
Reacher-v2_C := 0.0 0.01 0.1 1.0 10.0
Reacher-v2_NumTimeSteps := 100000
Reacher-v2_EvalFreq := 100

$(LOG_DIR):
	mkdir -p $@

$(IMG_DIR):
	mkdir -p $@

define f_constant =
$$(LOG_DIR)/$(1)/c_$(2)_$(3): |$$(LOG_DIR)
	mkdir -p $$@
	echo $$(GIT_TREE_HASH) > $$@/tree_hash.txt
	$$(JOB_SCHEDULER) python $$(MAIN_SCRIPT) --env $(1) --c $(2) --use_env_wrapper --num_time_steps $$($(1)_NumTimeSteps) --log_dir $$@ --verbose $$(VERBOSITY) --eval_freq $$($(1)_EvalFreq)

$(1)_constants = $$(foreach c,$$($(1)_C),$$(foreach rep,$$(REPS),$$(LOG_DIR)/$(1)/c_$$(c)_$$(rep)))
endef

$(foreach env,$(ENVS),$(foreach c,$($(env)_C),$(foreach rep,$(REPS),$(eval $(call f_constant,$(env),$(c),$(rep))))))

constants = $(foreach env,$(ENVS),$(foreach c,$($(env)_C),$(foreach rep,$(REPS),$(LOG_DIR)/$(env)/c_$(c)_$(rep))))

define f_constant_normalized =
$$(LOG_DIR)/$(1)/normalized_c_$(2)_$(3): |$$(LOG_DIR)
	mkdir -p $$@
	echo $$(GIT_TREE_HASH) > $$@/tree_hash.txt
	$$(JOB_SCHEDULER) python $$(MAIN_SCRIPT) --env $(1) --c $(2) --use_env_wrapper --num_time_steps $$($(1)_NumTimeSteps) --log_dir $$@ --verbose $$(VERBOSITY) --eval_freq $$($(1)_EvalFreq) --normalize_constraint

$(1)_constants_normalized = $$(foreach c,$$($(1)_C),$$(foreach rep,$$(REPS),$$(LOG_DIR)/$(1)/normalized_c_$$(c)_$$(rep)))
endef

$(foreach env,$(ENVS),$(foreach c,$($(env)_C),$(foreach rep,$(REPS),$(eval $(call f_constant_normalized,$(env),$(c),$(rep))))))

_constants_normalized = $(foreach env,$(ENVS),$(foreach c,$($(env)_C),$(foreach rep,$(REPS),$(LOG_DIR)/$(env)/c_$(c)_$(rep))))

define f_sac_constant =
$$(LOG_DIR)/$(1)/sac_c_$(2)_$(3): |$$(LOG_DIR)
	mkdir -p $$@
	echo $$(GIT_TREE_HASH) > $$@/tree_hash.txt
	$$(JOB_SCHEDULER) python $$(MAIN_SCRIPT) --env $(1) --c $(2) --use_env_wrapper --num_time_steps $$($(1)_NumTimeSteps) --log_dir $$@ --verbose $$(VERBOSITY) --eval_freq $$($(1)_EvalFreq) --solver SAC

$(1)_sac_constants = $$(foreach c,$$($(1)_C),$$(foreach rep,$$(REPS),$$(LOG_DIR)/$(1)/sac_c_$$(c)_$$(rep)))
endef

$(foreach env,$(ENVS),$(foreach c,$($(env)_C),$(foreach rep,$(REPS),$(eval $(call f_sac_constant,$(env),$(c),$(rep))))))

define f_geometric =
$$(LOG_DIR)/$(1)/a0$(2)_r$(3)_n$(4)_$(5): |$$(LOG_DIR)
	mkdir -p $$@
	echo $$(GIT_TREE_HASH) > $$@/tree_hash.txt
	$$(JOB_SCHEDULER) python $$(MAIN_SCRIPT) --env $(1) --a0 $(2) --r $(3) --use_env_wrapper --num_time_steps $$($(1)_NumTimeSteps) --log_dir $$@ --verbose $$(VERBOSITY) --n $(4) --eval_freq $$($(1)_EvalFreq)

$(1)_geometric = $$(foreach a0,$$(A0),$$(foreach r,$$(R),$$(foreach n,$(N),$$(foreach rep,$$(REPS),$$(LOG_DIR)/$(1)/a0$$(a0)_r$$(r)_n$$(n)_$$(rep)))))
endef

$(foreach env,$(ENVS),$(foreach a0,$(A0),$(foreach r,$(R),$(foreach n,$(N),$(foreach rep,$(REPS),$(eval $(call f_geometric,$(env),$(a0),$(r),$(n),$(rep))))))))

define f_augmented =
$$(LOG_DIR)/$(1)/a0$(2)_r$(3)_lr$(4)_n$(5)_$(6): |$$(LOG_DIR)
	mkdir -p $$@
	echo $$(GIT_TREE_HASH) > $$@/tree_hash.txt
	$$(JOB_SCHEDULER) python $$(MAIN_SCRIPT) --env $(1) --a0 $(2) --r $(3) --use_env_wrapper --num_time_steps $$($(1)_NumTimeSteps) --log_dir $$@ --verbose $$(VERBOSITY) --dual_learning_rate $(4) --n $(5) --eval_freq $$($(1)_EvalFreq)

$(1)_augmented = $$(foreach a0,$$(A0),$$(foreach r,$$(R),$$(foreach lr,$$(LR),$$(foreach n,$(N),$$(foreach rep,$$(REPS),$$(LOG_DIR)/$(1)/a0$$(a0)_r$$(r)_lr$$(lr)_n$$(n)_$$(rep))))))
endef

$(foreach env,$(ENVS),$(foreach a0,$(A0),$(foreach r,$(R),$(foreach lr,$(LR),$(foreach n,$(N),$(foreach rep,$(REPS),$(eval $(call f_augmented,$(env),$(a0),$(r),$(lr),$(n),$(rep)))))))))

$(IMG_DIR)/half_cheetah_dynamic.png: $(HalfCheetahDynamic_constants) | $(IMG_DIR)
	cd plot_scripts; python plot_half_cheetah_different_constants.py

$(IMG_DIR)/hopper_different_constants.png: $(Hopper-v2_constants) | $(IMG_DIR)
	cd plot_scripts; python plot_hopper_different_constants.py

.PHONY: docker_build
docker_build:
	$(info USERNAME is $(USERNAME))
	$(info USER_UID is $(USER_UID))
	docker build -t action_constrained_rl --build-arg USERNAME=$(USERNAME) --build-arg USER_UID=$(USER_UID) .

.PHONY: pytest
pytest:
	python -m pytest --import-mode=importlib -s

print-%  : ; @echo $* = $($*)

variable-%  : $($*)
	@echo $($*)
	$(MAKE) $($*)

clean-%  : $($*)
	@echo $($*)
	-rm -r $($*)

#.PHONY: print_all
#print_all:
#    $(foreach var,$(.VARIABLES),$(info $(var) = $($(var))))