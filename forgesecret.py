#Loading the libraries to be used 
from pyomo.environ import *
import matplotlib.pyplot as plt
import random 
import numpy as np
import pandas as pd
from pyomo.common.timing import TicTocTimer
from amplpy import modules

model = AbstractModel()
model.level = Param()
model.halfn = Param()
model.p = Param()
model.uba = Param() #upper bound of a     
model.i = Set()
model.j = RangeSet(1, model.halfn)
model.T = Param(model.i,model.j, within=Reals)
model.Mset = Param(model.i, within=Reals)

def initval(model,i):
    return random.randint(-model.uba,model.uba) #initialization of the variable
def init0(model,i):
    return(0)
def init1(model,i,j):
    return(0)
model.a = Var(model.i, bounds=(-model.uba,model.uba), within= Reals, initialize=initval)
model.b = Var(model.i, bounds=(-model.uba,model.uba) , within= Reals, initialize=initval)
model.q = Var(model.i, within=Integers,initialize=init0)
model.Pa = Var(model.i,model.j,  bounds=(-1,1), within=Integers,initialize=init1) 
model.Pb = Var(model.i,model.j,  bounds=(-1,1), within=Integers,initialize=init1) 

def rule_baT(model,i):
        return(np.sum([model.a[j]*model.T[j,i] for j in model.j ]) == model.q[i]*model.p + model.b[i])#b =aT    
model.C1   = Constraint(model.i,rule=rule_baT)

def rule_a(model,i):
    return(sum([model.Mset[j]*model.Pa[j,i] for j in model.j]) == model.a[i])#building a
model.Ca = Constraint(model.i, rule = rule_a)
def rule_b(model,i):
    return(sum([model.Mset[j]*model.Pb[j,i] for j in model.j]) == model.b[i])#building a
model.Cb = Constraint(model.i, rule = rule_b)

def rule_cPa(model,i):
    return((sum([(model.Pa[j,i])**2  for j in model.j]))==1)#each column exactly one nonzero elements  
model.cPa = Constraint(model.i, rule = rule_cPa)
def rule_cPb(model,i):
    return((sum([(model.Pb[j,i])**2 for j in model.j])) ==1)#each column has exactly one nonzero elements  
model.cPb = Constraint(model.i, rule = rule_cPb)
def rule_rPa(model,i):
    return((sum([(model.Pa[i,j])**2 for j in model.j])) ==1)#each row exactly one nonzero elements  
model.rPa = Constraint(model.i, rule = rule_rPa)
def rule_rPb(model,i):
    return((sum([(model.Pb[i,j])**2 for j in model.j])) ==1)#each row has exactly one nonzero elements  
model.rPb = Constraint(model.i, rule = rule_rPb)

def rule_OF(model):
    return sum(model.q[i] for i in model.i)

model.obj = Objective(rule=rule_OF, sense=maximize)
opt = SolverFactory(modules.find("highs"), solve_io="nl") #Couenne was good, ipopt did not give a result
#['bonmin', 'cbc', 'conopt', 'couenne', 'cplex', 'filmint', 'filter', 'ipopt', 'knitro', 'l-bfgs-b', 
#'lancelot', 'lgo', 'loqo', 'minlp', 'minos', 'minto', 'mosek', 'octeract', 'ooqp', 'path', 'raposa', 'snopt']

def forge_sk(filename = "toyexample.dat", store_at = "forgedsk.txt", verbose = False):
    timer = TicTocTimer()
    timer.tic('starting timer')
    instance = model.create_instance(filename)
    print('We just have built the instance, start solving stay tuned!')
    results = opt.solve(instance,keepfiles=True, logfile="my.log")

    
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        print('feasible')
    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        print('infeasible')
    else:
        print ('Solver Status:',  results.solver.status)
    print(value(instance.obj))
    a = [int(round(instance.a[i].value,0)) for i in instance.i]
    b = [int(round(instance.b[i].value,0)) for i in instance.i]
    sk = a+b
    T = [[instance.T[i,j] for j in instance.j] for i in instance.i]
    p = instance.p.value
    print('Verifying that indeed aT = b mod p \n This should be 1 if the sekret keys are valid:\n')
    print(prod(np.dot(a,T) %p == np.array(b)%p))
    #Write a and b in a file 
    np.savetxt(store_at, sk, fmt="%d", delimiter=",")
    dT = timer.toc('task 1')
    print("elapsed time: %0.1f s" % dT)
    if verbose:
        print('Value of a' + str(a))
        print('Value of b' + str(b))
        
        
#level 1 
forge_sk(filename = "level1example.dat", store_at = "forgedlevel1sk.txt", verbose = False)        
        
        
        
        
        
        
        
        
        



