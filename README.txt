Scripts for the 'Breaking FuLeeca with MIP' paper and the thesis.

We have the following scripts:

attackfuleeca.py: Python module that simulates the reference implementation for FuLeeca. Contains algorithms to generate a key pair, generate a signature,  verify a signature, transform the public key, and attack an instance with the NHF transformation. The notebook NewFuLeecainuse.ipynb shows how to use each function.

TextbookFuLeecaAttack.py: Python module that simulates the textbook FuLeeca. Contains algorithms to generate a key pair, generate a signature,  verify a signature, transform the public key, and attack an instance without the NHF transformation. The notebook OrigAttackNewFuLeecainuse.ipynb shows how to use each function.

TextbookFuLeecawithtNHF.py: Python module that simulates the textbook FuLeeca. Contains algorithms to generate a key pair, generate a signature,  verify a signature, transform the public key and attack an instance with the NHF transformation. The notebookTextbookFuLeecainuse.ipynb shows how to use each function.

GenerateFuLeecaKey.ipynb: creates a FuLeeca key pair and writes the public key to a CSV file. By default, a toy parameter set is used, but the parameters can be changed in the script.
FormatPublicKey.ipynb: From a public key file in CSV, create a .dat file that can be fed to Pyomo
 
Requires:
A full installation of AMPL and the other solvers in the local computer
Or running the Notebooks on Google Colab

Toy example
You can see the toy example presented in my thesis at the website https://sites.google.com/view/antsafuleecatoyexample. You can see the source code in the file ToyExampleSagemath.ipynb

 
