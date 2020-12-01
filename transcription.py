import numpy as np
import soundfile as sf
from scipy import signal
import nn_fac.nmf as nmf
import matplotlib.pyplot as plt
from numpy.matlib import repmat

# Read the song (you can use your own!)
the_signal, sampling_rate_local = sf.read('song/Jordu1.wav')
# Using the settings of the Attack-Decay transcription paper
frequencies, time_atoms, Y = signal.stft(the_signal[:, 0], fs=sampling_rate_local, nperseg=4096, nfft=8192, noverlap=4096 - 882)
time_step = time_atoms[1] #20 ms
freq_step = frequencies[1] #5.3 hz
time_atoms = time_atoms # ds scale
# Taking the power spectrogram
Y = np.abs(Y)**2
# adding some constant noise for avoiding zeros
Y = Y+1e-8
# Cutting silence, end song and high frequencies (>10k Hz)
cutf = 2000
cutt_in = 65
cutt_out = time_atoms.shape[0]
Y = Y[:cutf, cutt_in:cutt_out]

# get sizes
m, n = Y.shape

# Let's separate notes!
d = 30
out = nmf.nmf(Y, d, verbose=False, n_iter_max = 500)

# Normalize output
W = out[0]
H = out[1]
normsW = np.sum(W,0)
W = W*repmat(1/normsW, m, 1)
#H = np.diag(1/np.max(H,1))@H
H = np.diag(normsW)@H

# Post-treatment specific to one run
#permut = [7, 1, 3, 4, 5, 0, 6, 2]
#W = W[:,permut]
#H = H[permut,:]

# Only for slide show
#notes = ['noise','sol','do','re', 'mib', 'fa', 'sol', 'fa#']


# Printing W and H
plt.figure()
plt.imshow(W[:200, :], aspect='auto')
ticks = np.trunc(frequencies[0:200:10])
plt.yticks(range(0,200,10), ticks.astype(int))
plt.ylabel('Hz')
#plt.xticks(range(8),notes)


plt.figure()
for i in range(d):
    plt.subplot(d,1,i+1)
    plt.plot(H[i,:])
    plt.xticks([])
    plt.yticks([])
    if i==d-1:
        hop = 100
        ticks = np.trunc(10*time_atoms[cutt_in:cutt_out:hop])/10
        ticks_number = ticks.shape[0]
        plt.xticks(range(0,ticks_number*hop,hop), ticks)
    #plt.ylabel(notes[i])


# Printing Y
plt.figure()
plt.subplot(211)
plt.imshow(Y[:200,:])
yticks = np.trunc(frequencies[0:200:20])
plt.yticks(range(0,200,20), yticks.astype(int))
plt.ylabel('Hz')
hop = 100
xticks = np.trunc(10*time_atoms[cutt_in:cutt_out:hop])/10
ticks_number = xticks.shape[0]
plt.xticks(range(0,ticks_number*hop,hop), xticks)
plt.xlabel('time (s)')
plt.title('Y')
plt.subplot(212)
plt.imshow(np.sqrt(Y[:200,:]))
plt.yticks(range(0,200,20), yticks.astype(int))
plt.ylabel('Hz')
plt.xticks(range(0,ticks_number*hop,hop), xticks)
plt.xlabel('time (s)')
plt.title('sqrt(Y)')


plt.show()
