import attackfuleeca #importing the class
from datetime import datetime
import pandas as pd
attack = attackfuleeca.attack()
def attacklevel1(i):
    s = datetime.now()
    now = s.strftime("%Y%m%d%H%M%S") + 'iter' +str(i) #just in case they are solved very fast
    level = 1
    pkfilename = 'generatedpk/' + now +'T.csv'
    dat_name = 'dat_files/' + now
    skfilename = "originalsk/" + now + "level1sk.txt"
    forgedskname = "forgedsk/" + now + "level1sk.txt"
    infolist = [level, pkfilename, skfilename,forgedskname]
    #generate the key 
    attack.generate_key(level = level, pkfilename = pkfilename, skfilename = skfilename)
    T = attack.get_T(pkfilename)
    #format the key
    attack.T_to_dat(T, level=level, filename = dat_name)
    #perform the attack
    successlist = list(attack.forge_lin_sk(filename = dat_name + ".dat",solvername = 'gurobi', ampl = False))
    for k in successlist:
        infolist.append(k)
    attackdf = pd.DataFrame([infolist], columns =['Level', 'PKFilename',
                                                'SKorigfilename', 'ForgeSKFilename', 
                                                'Succes','ElapsedTime'])
    attackdf.to_csv("attackdf.csv",index=False,header=False,mode="a") 
    return(1) 
for i in range(5):
     attacklevel1(i)