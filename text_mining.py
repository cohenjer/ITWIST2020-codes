import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir
import nn_fac.nmf as nmf
from matplotlib import pyplot as plt
from numpy.matlib import repmat

# Importing data from the ./textes folder (Bring Your Own Text)
data = []
names = listdir('textes/')
for name in names:
    with open('./textes/'+name, 'r') as file:
        data.append(file.read().replace('\n', ' '))

n = len(names)

# Forming the word frequency matrix
vectorizer = TfidfVectorizer(stop_words='english')
Y = vectorizer.fit_transform(data)
words = np.array(vectorizer.get_feature_names())
#print(Y)
# Sadly nn_fac does not support sparse matrices yet...
Y = Y.todense()
# Normalizing FORBIDDEN, too much noise
#Y = np.multiply(Y, repmat(1/np.sum(Y, 0), n, 1))

# Factorizing Y with a rank-3 approximate NMF
rank = 3
out = nmf.nmf(Y, rank, verbose=True)

# Top 20 words per component
H = out[1]
top = []
topnum = 20
indices = np.zeros([rank, topnum])
for r in range(rank):
    temp = np.argsort(H[r, :])
    temp = np.flip(temp)
    indices[r, :] = temp[:topnum]
    top.append(words[temp.tolist()])

# plots
# Data
plt.figure()
plt.imshow(Y[:, :30])
plt.colorbar()
xticks = words[100:130]
plt.yticks(range(0, n), names)
plt.xticks(range(0, 30), xticks.tolist(), rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)

# Outputs
# W clusters
plt.figure()
plt.imshow(out[0]/np.max(out[0]))
plt.yticks(range(0, n), names)
plt.colorbar()

# H topics
plt.figure()
for r in range(rank):
    plt.subplot(1, rank, r+1)
    hr = H[r, :]
    ind = indices[r, :]
    plt.plot(hr[ind.astype(int)])
    xticks = words[ind.astype(int)]
    plt.xticks(range(0, topnum), xticks.tolist(), rotation='vertical')
    #plt.margins(0.2)
    plt.subplots_adjust(bottom=0.35)

plt.show()
