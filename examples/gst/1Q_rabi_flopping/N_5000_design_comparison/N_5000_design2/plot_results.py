from pathlib import Path
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import ionsim as sm
matplotlib.rcParams['text.usetex']=True 
style_path_data = '~/plot_style_data.txt'

if __name__ == '__main__':

    #N = 5
    # Base name for the directories (e.g., replicate_1, replicate_2...)
    DIR_PREFIX = ""

    FILE = 'num_circuits_study_Nboot_10.dat'
    data = np.loadtxt(FILE, unpack=True) 
    num_circuits = data[0]
    gate_error = data[1]
    std_err = data[2]

    #averaged_gate_set_error = np.zeros(15) 
    line_x = np.logspace(1., 1.6, 60) 
    offset = 5.0E-3 
    slope = -1.5

    # Compute slope from fit of the average 
    X = np.log(num_circuits)
    Y = np.log(gate_error)
    start_indx = 8
    end_indx = len(X) 
    coefficients = np.polyfit(X[start_indx:end_indx], Y[start_indx:end_indx], 1) 
    line_y = offset * (line_x) ** (slope)

    plt.style.use(style_path_data) 
    plt.figure(figsize = (5,5))
    plt.errorbar(num_circuits, gate_error, std_err, marker = 'o', linewidth = 0.5, markersize = 6, color = 'k', label='GST')
    plt.plot(line_x, line_y, linestyle = 'dashed', color='k', linewidth = 2., label= r'$m=' + str(slope) + '$')
    #plt.plot(np.exp(X), np.exp(coefficients[0]*X + coefficients[1]), linestyle = 'solid', color = 'k', linewidth = 1.5, label = r'Fit: $m = ' + str(np.round(coefficients[0].real, 3)) + '$')
    plt.plot(np.exp(X[start_indx:end_indx]), np.exp(coefficients[0]*X[start_indx:end_indx] + coefficients[1]), linestyle = 'solid', color = 'k', linewidth = 1.5, label = r'Fit: $m = ' + str(np.round(coefficients[0].real, 3)) + '$')
    plt.title(r'Gate Set Error vs. N circuits', fontsize = 14)
    plt.xlabel(r'Number of circuits', fontsize = 22)
    plt.ylabel(r'$||G - \bar{G}||$', fontsize = 20, rotation = 0, labelpad = 35)
    plt.xticks(fontsize = 12)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    #plt.savefig('X_pi8_err_N5000_shots_linear_fit.pdf', dpi=300)
    plt.show()



