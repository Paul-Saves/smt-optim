from smt_optim.acquisition_strategies.biego import BiEGO

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import math

from smt_optim.core import Problem
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.acquisition_strategies import MFSEGO
from smt_optim.acquisition_strategies.multiobj import MultiObj
from smt_optim.core import Sample
from smt_optim.core import ObjectiveConfig, DriverConfig
from smt_optim.core import Driver
from smt_optim.utils.constraints import compute_rscv

from custom_driver import CustomStopDriver

def Fonseca_Fleming1(x):
    return np.atleast_1d(1-np.exp(-np.sum((x-1/(np.sqrt(len(x))))**2)))

def Fonseca_Fleming2(x):
    return np.atleast_1d(1-np.exp(-np.sum((x+1/(np.sqrt(len(x))))**2)))

f1=Fonseca_Fleming1
f2=Fonseca_Fleming2

bounds=np.array([[-4, 4],[-4,4]])

surrogate=SmtAutoModel

max_iter=40

dim=2

n_init=2*dim+1

strat_kwargs={}


#Runs a default implementation of EGO on a bi-objective problem
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
    max_iter = max_iter,
    nt_init = n_init,
    verbose = True,
    scaling = True,
    seed=42,
)

strategy_kwargs = {
    "n_start":n_init
}

driver = Driver(prob_definition,opt_config,strategy=MultiObj)
#state=driver.optimize()

driver = Driver(prob_definition, opt_config, strategy=BiEGO)


state = driver.optimize()
print(state.export_as_dict())