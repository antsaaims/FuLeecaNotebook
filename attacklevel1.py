import attackfuleeca #importing the class
from datetime import datetime
import pandas as pd
attack = attackfuleeca.attack()
names = ["level1ex"+str(i) for i in range(10)]

def attacklevel1(name):
    '''attack the given instance with the given name'''
    level = 1
    pkfilename = 'generatedpk/' + name +'T.csv'
    dat_name = 'dat_files/' + name
    skfilename = "originalsk/" + name + "level1sk.txt"
    forgedskname = "forgedsk/" + name + "level1sk.txt"
    infolist = [level, pkfilename, skfilename,forgedskname]
    #generate the key 
    #attack.generate_key(level = level, pkfilename = pkfilename, skfilename = skfilename)
    #T = attack.get_T(pkfilename)
    #format the key
    #attack.T_to_dat(T, level=level, filename = dat_name)
    #perform the attack
    successlist = list(attack.forge_lin_sk(filename = dat_name + ".dat",solvername = 'gurobi', ampl = False, store_at = forgedskname))
    for k in successlist:
        infolist.append(k)        
    attackdf = pd.DataFrame([infolist], columns =['Level', 'PKFilename',
                                                'SKorigfilename', 'ForgeSKFilename', 
                                                'Succes','ElapsedTime'])
    attackdf.to_csv("attackdf.csv",index=False,header=False,mode="a") 
    return(1) 

for name in names:
     attacklevel1(name)