from smt_optim.acquisition_strategies.biego import BiEGO

import numpy as np

from smt_optim.core import Problem
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.acquisition_strategies.multiobj import MultiObj
from smt_optim.core import ObjectiveConfig, DriverConfig
from smt_optim.core import Driver

from smt_optim.benchmarks.registry import list_problems

bounds=np.array([[-4, 4],[-4,4]])

surrogate=SmtAutoModel

max_budget=100

n_accuracy=1000

L=list_problems(tags=["zdt"])

def run_benchmark(bproblem):
    name=bproblem.name
    print("Running BiEGO on benchmark problem",name)
    num_dim=bproblem.num_dim
    num_obj=bproblem.num_obj
    num_cstr=bproblem.num_cstr
    bounds=bproblem.bounds
    objective=bproblem.objective

    assert(num_obj==2)
    assert(num_cstr==0)

    f1=objective[0]
    f2=objective[1]


    n_init=2*num_dim+1

    obj_config1 = ObjectiveConfig(
        [f1],
        type="minimize",
        surrogate=surrogate,
    )

    obj_config2 = ObjectiveConfig(
        [f2],
        type="minimize",
        surrogate=surrogate,
    )

    prob_definition = Problem(
        obj_configs=[obj_config1,obj_config2],
        design_space=bounds,            # problem bounds
        costs=[1,1]
    )


    opt_config = DriverConfig(
        max_iter = max_budget - n_init,
        max_budget = max_budget,
        nt_init = n_init,
        verbose = True,
        scaling = True,
        seed=42,
    )

    strategy_kwargs = {
        "n_start":n_init,
        "n_accuracy":n_accuracy
    }

    driver = Driver(prob_definition, opt_config, strategy=BiEGO, strategy_kwargs=strategy_kwargs)


    state = driver.optimize()
    driver.strategy.show_pareto_front()

for bproblem in L:
    run_benchmark(bproblem)
