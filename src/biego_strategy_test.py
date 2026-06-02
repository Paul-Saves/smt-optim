from smt_optim.acquisition_strategies.biego import BiEGO

import numpy as np

from smt_optim.core import Problem
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.acquisition_strategies.multiobj import MultiObj
from smt_optim.core import ObjectiveConfig, DriverConfig
from smt_optim.core import Driver

from smt_optim.benchmarks.registry import list_problems

import matplotlib.pyplot as plt

bounds=np.array([[-4, 4],[-4,4]])

n_accuracy=1000

L=list_problems(tags=["zdt"])

surrogate=SmtAutoModel


def get_DoE(state):
    Y=state.dataset.export_as_dict()["obj"]
    D=state.dataset.export_as_dict()["x"]
    return (D,Y)

def Dominates(p,q):
    #Returns True if point p strictly dominates point q, else returns False
    return (p[0]<q[0] and p[1]<q[1])

def ParetoFront(D,Y):
    #Given a DoE (D,Y), returns the list of indices of non-dominated points, sorted by ascending value of f1
    t=len(D)
    front=[]
    for i in range(t):
        if all([not Dominates(q,Y[i]) for q in Y]):
            front.append((Y[i][0],i))
    front.sort()
    return [p[1] for p in front]

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

    max_budget=20*num_dim
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
    return state


data=[]

for bproblem in L:
    state=run_benchmark(bproblem)
    data.append((bproblem.name,state))

_,axs=plt.subplots(3,3)

for i in range(3):
    for j in range(3):
        index = i*3+j
        D,Y=get_DoE(data[index][1])
        pareto_points = [(D[i],Y[i]) for i in ParetoFront(D,Y)]
        
        ax=axs[i][j]
        ax.scatter([p[1][0] for p in pareto_points],[p[1][1] for p in pareto_points])
        ax.title.set_text(data[index][0])

plt.show()
