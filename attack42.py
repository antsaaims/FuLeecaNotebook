import attackfuleeca #importing the class
from datetime import datetime
import pandas as pd

for i in range(10):
    s = datetime.now()
    now = s.strftime("%Y%m%d%H%M%S")
    level = 42
    pkfilename = 'generatedpk/' + now +'T.csv'
    dat_name = 'dat_files/' + now + 'dat.csv'
    skfilename = "originalsk/" + now + "toysk.txt"
    forgedskname = "forgedsk/" + now + "forgedsk.txt"
    infolist = [level, pkfilename, skfilename,forgedskname]
    #generate the key 
    attack.generate_key(level = level, pkfilename = pkfilename, skfilename = skfilename)
    T = attack.get_T(pkfilename)
    #format the key
    attack.T_to_dat(T, level=level, filename = dat_name)
    #perform the attack
    successlist = list(attack.forge_lin_sk(solvername = 'gurobi', ampl = False))
    for k in successlist:
        infolist.append(k)
    attackdf = pd.DataFrame([infolist], columns =['Level', 'PKFilename',
                                                'SKorigfilename', 'ForgeSKFilename', 
                                                'Succes','ElapsedTime'])
    attackdf.to_csv("attackdf.csv",index=True,header=False,mode="a") 