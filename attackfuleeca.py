# This is a python class 
# Created on 9/28/2023 by Antsa 
# Contains all the functions that are needed to generate public and secret key, prepare them for the attack and run the attack
# Please do not change this file, call it using from attackfuleeca import* to use all the functions
import subprocess
import pandas as pd
import numpy as np
import galois
import random as rnd
from scipy.linalg import circulant
from functools import reduce
from pyomo.environ import *
import matplotlib.pyplot as plt
from pyomo.common.timing import TicTocTimer
from math import exp, floor
from amplpy import modules
modules.install('highs')
modules.install('gurobi')
print('Solver you already have')
print(modules.installed())
def install(module, tool = 'pip'):
    '''Check if a python library is install and intsall it'''
    subprocess.check_call([tool, 'install', module])
    print(f"The module {module} was installed")
# def requirements():    
#     for module in ['pandas', 'galois', 'random', 'scipy','functools', 'matplotlib', 'pyomo', 'numpy','amplpy']:
#         install(module)
#     print('If you did not intsall your amplpy licence yet,\n please visit https://ampl.com/ce and request a community licence\n after that tyoe "amplpy.modules activate <your license-uuid>"\n You may also find the ampl.modules documentation here https://dev.ampl.com/ampl/python/modules.html useful')
#     print("If you plan to use Gurobi, you need top request a licence at gurobi.com")
#     return(None)
class attack(): 

    def pw(self,x,p =65521, beta =0.021240595234768,Z = 94.1628598983449): #probability of x occuring
        '''Given x, how many times does x occures'''
        return (1/Z)*exp(-1*beta*min(x, p-x));

    tau=0.25 #estimate of tau, which decides what goes in TLS

    def NumberInTLS(self,x, halfn,p =65521, beta =0.021240595234768,Z = 94.1628598983449,tau=0.25):
        '''Given an element in the field,how many of it will be in the TLS'''
        a= self.pw(x,p,beta,Z)*halfn
        if a >= 1:
            b=floor(a)
        else:
            if a >= tau:
                b=1
            else:
                b=0
        return b

    def MakeTLS(self,halfn,p =65521, beta =0.021240595234768,Z = 94.1628598983449,tau=0.25):
        '''Given the value of halfn, generate the typical Lee Set'''
        TLS=[]
        for i in range(p):
            TLS= TLS + [i for j in range(self.NumberInTLS(i,halfn,p,beta,Z,tau))]
        if len(TLS)<halfn:
            TLS =TLS + [0 for i in range(halfn-len(TLS))]
        return TLS
    SmallInstancesHalfn = [42,83,165,331,659,991,1319]#the ones that are interesting

    #Key Generation ALgorithm in Python
     # choose 0 for toy example, 1,3,5 for each corresponding NIST security level
    #Key Generation

    def generate_key(self,level =0, verbose = False, pkfilename = 'T.csv', skfilename = "toysk.txt"):
        #Setting the parameters corresponding to the security level
        if level == 0:
            p = 5
            mset = [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4]        
        elif level ==1:
            p = 65521
            mset = mset1
        elif level in SmallInstancesHalfn:
            p = 65521
            mset =  MakeTLS(level)
        else: 
            print("That level does not exists in this module")
            return(None)
        #Define typical set based on param set
        GF = galois.GF(p)
        Mset = np.array([min(abs(x - p),x) for x in mset])
        halfn = len(Mset) 
        uba = (p-1)//2 #upper bound of a 
        def poly_transform_to_two_side(self,poly):
            '''Transforms one sided shuffled
            typical set to two sided 
            by flipping random entries''' 
            howtoflip = np.array([rnd.choice([1,-1]) for i in range(len(poly))])
            return(np.array(poly)*howtoflip)
        def poly_sample_from_typical_set(self):
            '''Generate an element of the typical Lee set'''
            #Init and shuffle range array
            res = reduce(lambda acc, _: np.random.permutation(acc),
                     range(len(Mset)), np.array(Mset))
            poly = res.tolist()
            #Use random bits to flip signs of coeffs. in key    
            return(self.poly_transform_to_two_side(poly))
        b_orig = self.poly_sample_from_typical_set()
        stop = 0
        while not stop:
            a_orig = self.poly_sample_from_typical_set()
            A_orig = GF(circulant([x%p for x in a_orig]).transpose())
            if np.linalg.det(A_orig) !=0:
                stop = 1           

        B_orig = GF(circulant([x%p for x in b_orig]).transpose())
        # value for T
        T = np.dot(np.linalg.inv(A_orig), B_orig) #Numpy does not know * as a matrix mutliplication
        T = np.array(T) ## back to R, as we do not have any notion of order in Fp
        if verbose:
            print("Alice's a value: " + str(a_orig))
            print("Alice's b value: " + str(b_orig))
            print("Alice's T value: ")
            print(T)
        # If you are gonna publish your public key, use this 
        #Public key
        np.savetxt(pkfilename, T, fmt="%d", delimiter=",")
        #Private key
        sk = list(a_orig)+list(b_orig)
        np.savetxt(skfilename, sk, fmt="%d", delimiter=",")
        print('This should be 1 if the sekret keys are valid')
        print(np.prod(np.dot(a_orig,T) %p == b_orig%p))
        return(None)

    #Formatting
    #use this if you obtained the T value as a file.
    def get_T(self,T_csv):
        '''Given a csv/text file containing T, output T as an numpy array'''
        T =  np.loadtxt(T_csv, delimiter=",",dtype = 'uint8') 
        return(T)        
    def T_to_dat(self,T,level=0,filename = 'toyexample'):
        '''Given an array/list of T, create a .dat file that is compatible with pyomo'''
            #Setting the parameters corresponding to the security level
        if level == 0:
            p = 5
            halfn = 11
            mset = [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4]
        elif level ==1:
            p = 65521
            halfn = 659
            mset = mset1
        elif level in SmallInstancesHalfn:
            p = 65521
            mset =  MakeTLS(level)
            halfn = level
        else: 
            print("That level does not exists in this module")
            return(None)    
        uba = (p-1)//2 #upper bound of a 
        Mset = np.array([min(abs(x - p),x) for x in mset])
        #This code works but not optimized
        s = 'param level:=' + str(level)
        s = s + '; \nparam p:= ' +str(p) 
        s = s + '; \nparam uba:= ' +str(uba)
        s = s + '; \nparam halfn:= ' +str(halfn) 

        # s = s + '; \nparam Mset:'
        # s = s+ ' '.join([str(i+1) for i in range(halfn)])  + ':=\n'
        s = s+ '; \nparam: i:\n'
        s =s+ '       Mset:='

        for i in range(halfn):
            s = s +  '\n      '+str(i+1) + ' ' + str(Mset[i]) 

        #s = s+ '           '+ ' '.join([str(Mset[i]) for i in range(halfn)])
        s = s + '; \nparam T:  '
        s = s+' '.join([str(i+1) for i in range(halfn)])  + ':=\n'
        s = s+  '\n'.join(( [ '        '+str(i+1) +' '+ str(T[i])[1:-1] for i in range(halfn)]))
        s = s+';'  
        #np.savetxt(filename + '.dat', s , fmt='%s')
        with open(filename+'.dat',"w") as f:
            f.write(s)
        print('Task done, see your file at ' + filename +'.dat')
        return(s)

    ##attack
    def __init__(self): 
        #Level 1 FuLeeca From Actual authors
        self.mset1 = [65363, 65364, 65365, 65366, 65367, 65368, 65369, 65370, 65371, 65372, 65373, 65374, 65375, 65376, 65377, 65378, 65379, 65380, 65381, 65382, 65383, 65384, 65385, 65386, 65387, 65388, 65389, 65390, 65391, 65392, 65393, 65394, 65395, 65396, 65397, 65398, 65399, 65400, 65401, 65402, 65403, 65404, 65405, 65406, 65407, 65408, 65409, 65410, 65411, 65412, 65413, 65414, 65415, 65416, 65417, 65418, 65419, 65420, 65421, 65422, 65423, 65424, 65425, 65426, 65427, 65428, 65429, 65430, 65431, 65432, 65433, 65434, 65435, 65436, 65437, 65438, 65439, 65440, 65441, 65442, 65443, 65444, 65445, 65446, 65447, 65448, 65449, 65450, 65451, 65452, 65453, 65454, 65455, 65456, 65457, 65458, 65459, 65460, 65461, 65462, 65463, 65463, 65464, 65464, 65465, 65465, 65466, 65466, 65467, 65467, 65468, 65468, 65469, 65469, 65470, 65470, 65471, 65471, 65472, 65472, 65473, 65473, 65474, 65474, 65475, 65475, 65476, 65476, 65477, 65477, 65478, 65478, 65479, 65479, 65480, 65480, 65481, 65481, 65482, 65482, 65482, 65483, 65483, 65483, 65484, 65484, 65484, 65485, 65485, 65485, 65486, 65486, 65486, 65487, 65487, 65487, 65488, 65488, 65488, 65489, 65489, 65489, 65490, 65490, 65490, 65491, 65491, 65491, 65492, 65492, 65492, 65493, 65493, 65493, 65494, 65494, 65494, 65495, 65495, 65495, 65495, 65496, 65496, 65496, 65496, 65497, 65497, 65497, 65497, 65498, 65498, 65498, 65498, 65499, 65499, 65499, 65499, 65500, 65500, 65500, 65500, 65501, 65501, 65501, 65501, 65502, 65502, 65502, 65502, 65503, 65503, 65503, 65503, 65504, 65504, 65504, 65504, 65505, 65505, 65505, 65505, 65506, 65506, 65506, 65506, 65506, 65507, 65507, 65507, 65507, 65507, 65508, 65508, 65508, 65508, 65508, 65509, 65509, 65509, 65509, 65509, 65510, 65510, 65510, 65510, 65510, 65511, 65511, 65511, 65511, 65511, 65512, 65512, 65512, 65512, 65512, 65513, 65513, 65513, 65513, 65513, 65514, 65514, 65514, 65514, 65514, 65514, 65515, 65515, 65515, 65515, 65515, 65515, 65516, 65516, 65516, 65516, 65516, 65516, 65517, 65517, 65517, 65517, 65517, 65517, 65518, 65518, 65518, 65518, 65518, 65518, 65519, 65519, 65519, 65519, 65519, 65519, 65520, 65520, 65520, 65520, 65520, 65520, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.model = AbstractModel()
        self.model.level = Param()
        self.model.halfn = Param()
        self.model.p = Param()
        self.model.uba = Param() #upper bound of a     
        self.model.i = Set()
        self.model.j = RangeSet(1, self.model.halfn)
        self.model.T = Param(self.model.i,self.model.j, within=Reals)
        self.model.Mset = Param(self.model.i, within=Reals)

        def initval(model,i):
            return random.randint(-self.model.uba,self.model.uba) #initialization of the variable
        def init0(model,i):
            return(0)
        def init1(model,i,j):
            return(0)
        self.model.a = Var(self.model.i, bounds=(-self.model.uba,self.model.uba), within= Reals, initialize=init0)
        self.model.ap = Var(self.model.i, bounds=(0,self.model.uba), within= Reals, initialize=init0)
        self.model.am = Var(self.model.i, bounds=(-self.model.uba,0), within= Reals, initialize=init0) # am
        self.model.b = Var(self.model.i, bounds=(-self.model.uba,model.uba) , within= Reals, initialize=init0)
        self.model.bp = Var(self.model.i, bounds=(0,self.model.uba) , within= Reals, initialize=init0)
        self.model.bm = Var(self.model.i, bounds=(-self.model.uba,0) , within= Reals, initialize=init0)
        self.model.q = Var(self.model.i, within=Integers,initialize=init0)
        self.model.Pap = Var(self.model.i,self.model.j,  bounds=(0,1), within=Integers,initialize=init1) 
        self.model.Pbp = Var(self.model.i,self.model.j,  bounds=(0,1), within=Integers,initialize=init1) 
        self.model.Pam = Var(self.model.i,self.model.j,  bounds=(0,1), within=Integers,initialize=init1) 
        self.model.Pbm = Var(self.model.i,self.model.j,  bounds=(0,1), within=Integers,initialize=init1) 

        def rule_baT(model,i): #b =aT
                return(np.sum([model.a[j]*model.T[j,i] for j in model.j ]) == model.q[i]*model.p + model.b[i])#b =aT    
        self.model.C1   = Constraint(self.model.i,rule=rule_baT)

        def rule_am(model,i): #am = -Mset*Pam
            return(sum([-model.Mset[j]*model.Pam[j,i] for j in model.j]) == model.am[i])#building a
        self.model.Cam = Constraint(self.model.i, rule = rule_am)
        def rule_ap(model,i): #ap = Mset*Pap
            return(sum([model.Mset[j]*model.Pap[j,i] for j in model.j]) == model.ap[i])#building a
        self.model.Cap = Constraint(self.model.i, rule = rule_ap)

        def rule_sum_a(model,i): #ap = Mset*Pap
            return(model.am[i]+ model.ap[i]== model.a[i])#building a
        self.model.sum_a = Constraint(self.model.i, rule = rule_sum_a)

        def rule_bm(model,i): #am = -Mset*Pam
            return(sum([-model.Mset[j]*model.Pbm[j,i] for j in model.j]) == model.bm[i])#building a
        self.model.Cbm = Constraint(self.model.i, rule = rule_bm)
        def rule_bp(model,i): #ap = Mset*Pap
            return(sum([model.Mset[j]*model.Pbp[j,i] for j in model.j]) == model.bp[i])#building a
        self.model.Cbp = Constraint(self.model.i, rule = rule_bp)

        def rule_sum_b(model,i): #ap = Mset*Pap
            return(model.bm[i]+ model.bp[i]== model.b[i])#building a
        self.model.sum_b = Constraint(self.model.i, rule = rule_sum_b)



        def rule_cPa(model,i):
            return((sum([(model.Pam[j,i])  for j in model.j])) + (sum([(model.Pap[j,i])  for j in model.j]))==1)#each column exactly one nonzero elements  
        self.model.cPa = Constraint(self.model.i, rule = rule_cPa)

        def rule_cPb(model,i):
            return((sum([(model.Pbm[j,i]) for j in model.j])) + (sum([(model.Pbp[j,i]) for j in model.j]))==1)#each column has exactly one nonzero elements  
        self.model.cPb = Constraint(self.model.i, rule = rule_cPb)

        def rule_rPa(model,i):
            return((sum([(model.Pam[i,j]) for j in model.j])) +(sum([(model.Pap[i,j]) for j in model.j])) ==1)#each row exactly one nonzero elements  
        self.model.rPa = Constraint(self.model.i, rule = rule_rPa)
        def rule_rPb(model,i):
            return((sum([(model.Pbm[i,j]) for j in model.j])) + (sum([(model.Pbp[i,j]) for j in model.j]))==1)#each row has exactly one nonzero elements  
        self.model.rPb = Constraint(self.model.i, rule = rule_rPb)


        def rule_OF(model):
            return sum(model.q[i] for i in model.i)

        self.model.obj = Objective(rule=rule_OF, sense=maximize)
    def forge_lin_sk(self,filename = "toyexample.dat", store_at = "forgedsk.txt", solvername = 'highs', ampl = True, verbose = False):
        if ampl==True:
            opt = SolverFactory(modules.find(solvername), solve_io="nl") #Couenne was good, ipopt did not give a result(always max iteration)
        else:
            opt = SolverFactory(solvername)
        if solvername == 'gurobi':
            # put some option here to focus on feasiblility
            opt.options['MIPFocus'] =1 #focus on feasibility
            opt.options['ZeroObjNodes'] =5  #we want to run an heuristic search if the model did not find anything
            opt.options['IntegralityFocus'] =1 #Please take the integrality seriously
            opt.options['SubMIPCuts'] =2
        timer = TicTocTimer()
        timer.tic('starting timer')
        instance = self.model.create_instance(filename)
        print('We just have built the instance, start solving stay tuned!')
        results = opt.solve(instance,keepfiles=True, logfile="my.log")


        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            print('feasible')
        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            print('infeasible')
        else:
            print ('Solver Status:',  results.solver.status)
        print(value(instance.obj))
        a = [int(round(instance.a[i].value,0))%p for i in instance.i]
        b = [int(round(instance.b[i].value,0))%p for i in instance.i]
        sk = a+b
        T = [[instance.T[i,j] for j in instance.j] for i in instance.i]
        p = instance.p.value
        print('Verifying that indeed aT = b mod p \n This should be 1 if the sekret keys are valid:\n')
        prodat = np.dot(a,T) %p
        isokay = prod(prodat == np.array(b))
        print(isokay)
        #Write a and b in a file 
        np.savetxt(store_at, sk, fmt="%d", delimiter=",")
        dT = timer.toc('task 1')
        print("elapsed time: %0.1f s" % dT)
        if verbose:
            print('Value of a' + str(a))
            print('Value of b' + str(b))
            print('Value of aT' + str(prodat))
        return([isokay,dT])
    def __str__(self):
         return("Hello, I am the python class used to attack FuLeeca,\n I can be used to generate a key pair for FuLeeca,\n I also can load a public key from a file,\n I format the public key so that it can be used in Pyomo,\n, I use Pyomo to attack the FuLeeca cryptosystem and you got to choose which solver to use!")




