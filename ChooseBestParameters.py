import numpy as np  
from collections import defaultdict  
rezultate=defaultdict(list)  
import scipy.io as sio  
mat = sio.loadmat(r'C:\Users\Z\Downloads\baza de date\bci\testing\processed\checking\New folder\corectformat\100Cu400.mat')  
r=mat.get('X');  
y=mat.get('y');  
y=np.ravel(y);  
k=0;  
#Se clasifica pe rand fiecare voluntar și se salvează rezultatul într-un dicționar
for x in range(5):  
   X=r[:,:,x];  
   rezultate[k].append([(classify_functions(X,y))])  
