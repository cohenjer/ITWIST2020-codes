import numpy as np
import nn_fac
import scipy.io
import matplotlib.pyplot as plt

# Data from Rasmus Bro's open database
data = scipy.io.loadmat('fluodata/data.mat', squeeze_me=True)
Y = data['data']
Y = np.transpose(Y,[2,1,0])

# Info from the data repo
lem_scale = range(240,301) #mode 0 (typo on the website, 240 instead of 250)
lex_scale = range(250,451) #mode 1

# Computing the approximate NTF
rank = 3
out = nn_fac.ntf.ntf(Y, rank, n_iter_max = 1000, verbose = True, normalize=[True, True, False])

# Plots
plt.figure()
plt.subplot(131)
plt.plot(lem_scale, out[0])
plt.title('Emission spectra')
plt.xlabel('Wavelength (nm)')
plt.subplot(132)
plt.plot(lex_scale, out[1])
plt.title('Excitation spectra')
plt.xlabel('Wavelength (nm)')
plt.subplot(133)
plt.plot(out[2])
plt.title('Relative concentrations')

plt.show()
