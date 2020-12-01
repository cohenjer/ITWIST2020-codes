import numpy as np
from matplotlib import pyplot as plt
import nn_fac

# Importing image with RBG colors
Aim = plt.imread('./homer/homer_cheap_fast.jpg')
Aim = np.array(Aim)/255

# converting to matrix
n, m, p = Aim.shape
A = np.reshape(Aim, [n*m, p])
At = A.T # the color x pix matrix

# Problem 1: computing nnls
PaintPots = np.transpose(np.array([[112, 236, 76], [242, 85, 45]])/255)
out1 = nn_fac.nnls.hals_nnls_acc(PaintPots.T@At, PaintPots.T@PaintPots, np.random.rand(2, n*m), delta=1e-8)
Aest1 = np.minimum(PaintPots@out1[0], 1)
Aeim1 = np.reshape(Aest1.T, [n, m, p])

# Problem 2: computing the nmf
out2 = nn_fac.nmf.nmf(A, 2)

# Reconstructing the image
Aest2 = np.minimum(out2[0]@out2[1], 1)
Aeim2 = np.reshape(Aest2, [n, m, p])

# note: not a unique solution
print('The two colors used are ', out2[1])

# Error computing
err = np.sqrt(np.sum((A - Aest2)**2))/np.linalg.norm(A)
