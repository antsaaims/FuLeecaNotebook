Scripts for the 'Breaking FuLeeca with MIP' paper.

We have the following scripts:

GenerateFuLeecaKey.ipynb: creates a FuLeeca key pair writes the public key to a csv file. By default a toy parameter set is used, but the parameters can be changed in the script.
FormatPublicKey.ipynb: From a public key file in csv, create a .dat file that can be fed to Pyomo

Requires:
A full installation of AMPL and the other solvers in the local computer
Or running the Notebooks on Google Colab

Demonstration of the full attack
The "Attack_demo" folder contains two sage scripts that demonstrate the full attack.

Running Full_Attack.sage compiles t 
Once a solution is found it completes the key recovery attack, writes the recovered secret key to a file, and compares AT to B to verify that the attack was succesful.