import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt
import nn_fac

# Importing image with RBG colors
Aim = plt.imread('./homer/homer_eating.jpeg')
Aim = Aim+ 10**-6  # To avoid zeros in the data without having to remove them by hand
Aim = np.array(Aim)/255

# converting to matrix
n, m, p = Aim.shape
A = np.reshape(Aim, [n*m, p])
At = A.T # the color x pix matrix

# Setting some arbitrary values for Paint Pots
PaintPots = np.array([[0.36530265, 0.20134814],
       [0.29883148, 0.41997252],
       [0.33586587, 0.37867934]])

# %%Problem 2 uniqueness: several approximate rank2 NMFs

# First, making the data nonnegative rank 2 using NNLS
out1 = nn_fac.nnls.hals_nnls_acc(PaintPots.T@At, PaintPots.T@PaintPots, np.zeros([2,n*m]), delta=1e-8)
X = out1[0]
Aest1 = PaintPots@out1[0]

norms_data = np.sum(Aest1,0)
norms_X = np.sum(X,0)

Aest1pn = np.copy(Aest1)
Xpn = np.copy(X)
# Normalizations for second plots
Aest1 = Aest1*np.matlib.repmat(1/norms_data,3,1)
X = X*np.matlib.repmat(1/norms_X,2,1)

# %% Showing the data in 3d and 2d
subsamp = np.random.permutation(n*m)
subsamp = subsamp[:50000]
plt.figure()
plt.subplot(223) #unnormalized
plt.scatter(Xpn[0,subsamp],Xpn[1,subsamp], marker='o')
plt.plot([1,0],[0,1],c='r',linewidth=2)
plt.xlabel('x_{1n}')
plt.ylabel('x_{2n}')
plt.subplot(224) #normalized
plt.scatter(X[0,subsamp],X[1,subsamp], marker='o')
plt.plot([1,0],[0,1],c='r',marker = 'o', linestyle = '')
plt.subplot(222, projection='3d')
plt.plot(Aest1[0,subsamp],Aest1[1,subsamp],Aest1[2,subsamp], linestyle='', marker='o', ms=1)#, s=10)
plt.plot(PaintPots[0,:],PaintPots[1,:],PaintPots[2,:], c='r', linewidth=1)
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0,1])
axes.set_zlim([0,1])
plt.title('l1 Normalized')
plt.subplot(221, projection='3d')
plt.plot(Aest1pn[0,subsamp],Aest1pn[1,subsamp],Aest1pn[2,subsamp], linestyle='', marker='o', ms=1)#, s=10)
plt.plot(PaintPots[0,:],PaintPots[1,:],PaintPots[2,:], c='r', linewidth=1)
plt.xlabel('Red')
plt.ylabel('Green')
plt.title('Unnormalized')
#plt.zlabel('Blue')
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0,1])
axes.set_zlim([0,1])
axes.set_zlabel('Blue')
plt.show()

# %% Exact rank 2 decomposition with HALS

# Using the paintpots image with 2 colors plus addings some noise
A_r2ish =np.copy(Aest1)
A_r2im = A_r2ish*np.matlib.repmat(norms_data,3,1)
A_r2im = np.reshape(A_r2im.T, [n, m, p])
plt.figure()
plt.imshow(A_r2im)
plt.show()

color = []
colorpn = []
Z = []
costs = []
N = 9 #<16
# Computing a rank-2 approximate NMF 9 times
for i in range(N):
    # Initialization
    #randints = [np.random.randint(n*m) for i in range(2)]
    #W0 = A_r2ish[:,randints]  # with the data
    W0 = np.abs(np.random.randn(p, 2))  # Initial scatter
    W0 = W0*np.matlib.repmat(1/np.sum(W0,0),3,1)
    # Init with our W, otherwise first itertion is on H
    temp0 = nn_fac.nnls.hals_nnls_acc(W0.T@A_r2ish, W0.T@W0, np.random.rand(2,n*m), delta=1e-8)
    H0 = temp0[0]
    #H0 = np.random.rand(2, n*m)  # X
    # NMF
    out = nn_fac.nmf.nmf(A_r2ish, 2, init='custom', U_0=W0, V_0=H0 ,n_iter_max=1000, verbose=True)
    # Post l1 normalization
    H = np.copy(out[1])
    W = np.copy(out[0])
    colorpn.append(W)
    # putting W on the simplex
    W = W*np.matlib.repmat(1/np.sum(W,0),3,1)
    # Running a NNLS to find the scores H
    temp = nn_fac.nnls.hals_nnls_acc(W.T@A_r2ish, W.T@W, H, delta=1e-8)
    H = temp[0]
    # Storage
    color.append(W)
    Atemp = W@H
    Z.append(Atemp)
    costs.append(np.linalg.norm(Atemp-A_r2ish,'fro')/np.linalg.norm(A_r2ish, 'fro')*100)

plt.figure()
plt.subplot(111, projection='3d')
plt.plot(PaintPots[0,:],PaintPots[1,:],PaintPots[2,:], c='r', linewidth=1)
cs=['k', 'm', 'y', 'g']
for i in range(N):
    plt.plot(color[i][0,:], color[i][1,:], color[i][2,:], c=cs[i%4],linestyle = '', marker='s', ms=5)
axes = plt.gca()
plt.show()

plt.figure()
for i in range(N):
    plt.subplot(np.sqrt(N),np.sqrt(N),i+1)
    A_r2est = np.minimum(Z[i],1)*np.matlib.repmat(norms_data,3,1)
    A_r2ime = np.reshape(A_r2est.T, [n, m, p])
    plt.imshow(A_r2ime)

plt.figure()
for i in range(N):
    plt.subplot(np.sqrt(N),np.sqrt(N),i+1)
    plt.imshow(np.reshape(color[i],[1,2,p]))
    plt.title(str(int(np.floor(costs[i]*1e6)))+'x10^{-6}')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

plt.figure()
plt.imshow(np.reshape(PaintPots,[1,2,p]))
plt.show()
