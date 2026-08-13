#import ionsim as ism
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
matplotlib.rcParams['text.usetex']=True 
style_path_data = '~/plot_style_data.txt'



datafile = 'negative_log_likelihood.dat'  

data = np.loadtxt(datafile,unpack=True)
iterations = data[0]
neg_ll = data[1]

plt.style.use(style_path_data)
plt.figure(figsize=(6,6))
plt.plot(iterations, neg_ll, marker = 'o', markersize = 6, color = 'k') 
plt.xlabel('Iterations', fontsize = 16)
plt.ylabel(r'$-\log(\mathcal{L})$', fontsize = 16, rotation=0, labelpad = 25)
#plt.legend()
plt.show()
