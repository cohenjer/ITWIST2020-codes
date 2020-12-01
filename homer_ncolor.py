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
colors = [[112,236,76], [242,85,45], [103,250,131],[201,104,234],[228,219,51]]
PaintPots = np.transpose(np.array(colors)/255)
out1 = nn_fac.nnls.hals_nnls_acc(PaintPots.T@At, PaintPots.T@PaintPots, np.random.rand(5, n*m), delta=1e-8)

# Post-processing
x_est = out1[0]
Aest1 = np.minimum(PaintPots@out1[0], 1)
Aeim1 = np.reshape(Aest1.T, [n, m, p])
# Counting the number of nonzeros in the solution
x_l0 = sum(x_est != 0, 0)
x_im = np.reshape(x_l0, [n,m])

# plotting
plt.figure()
plt.subplot(221)
plt.imshow(Aim)
plt.subplot(222)
plt.imshow(np.reshape(np.array(colors),[1,5,3]))
plt.subplot(223)
plt.imshow(Aeim1)
plt.subplot(224)
plt.imshow(x_im)
plt.colorbar()
plt.show()
