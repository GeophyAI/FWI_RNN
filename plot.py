import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

# d = np.load("/mnt/others/DATA/Inversion/RNN_Marmousi/data/data.npy").squeeze()
# no = 5
# vmin,vmax=np.percentile(d[no], [2, 98])
# plt.imshow(d[no], vmin=vmin, vmax=vmax, cmap=plt.cm.gray, aspect='auto')
# plt.show()
# print(d.shape)
# exit()

"""
Model cut
"""
#d = np.load("/mnt/others/DATA/Inversion/RNN_Marmousi/inversion/vel3.00freq_29epoch.npy").squeeze()
# ori = np.load("/mnt/others/DATA/Inversion/RNN_Hessian/velocity/Ori_20_20_173x561_500msea.npy").squeeze()[:,50:-50]
# ori = ori[15:36, 250:301]
# print(ori.shape)
# np.save("/mnt/others/DATA/Inversion/RNN_Hessian/velocity/Ori_20_20_32x51_part.npy", ori)
# inv = np.load("/mnt/others/DATA/Inversion/RNN_Hessian/inversion/vel5.00freq_07epoch.npy").squeeze()[60:-60,60+50:-60-50]
# #d = np.load("/mnt/others/DATA/Inversion/RNN_Marmousi/velocity/Init_20_20_173x561_500msea.npy")

"""
Results
"""
ori = np.load("/mnt/others/DATA/Inversion/RNN_Marmousi/velocity/Ori_20_20_173x561_500msea.npy")
inv = np.load("/mnt/others/DATA/Inversion/RNN_Marmousi/inversion/vel8.00freq_29epoch.npy")[10:-10,10:-10]
smm = np.load("/mnt/others/DATA/Inversion/RNN_Marmousi/velocity/Init_20_20_173x561_500msea.npy")
#inv-=smm
fig, axes = plt.subplots(1, 3)
(vmin, vmax) = (1500, 5500)
axes[0].imshow(ori, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
axes[1].imshow(inv, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
axes[2].imshow(smm, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')

plt.show()
# print(d.shape)


# grad = np.load("/mnt/others/DATA/Inversion/RNN_Hessian/grad/grad_0hot.npy")
# grad2 = np.load("/mnt/others/DATA/Inversion/RNN_Hessian/grad/grad_2hot.npy")
# grad = grad2 -grad1
# shape = grad.shape
# vmin,vmax=np.percentile(grad, [2, 98])
# plt.imshow(grad, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
# plt.show()
"""
Hessian plot
"""
# hessian = []
# import os, sys, glob
# for file in sorted(glob.glob(os.path.join("/mnt/others/DATA/Inversion/RNN_Hessian/hessian", "*"))):
#     hessian.append(np.load(file).flatten())
# hessian = np.array(hessian)
#
# vmin,vmax=np.percentile(hessian, [2, 98])
# plt.imshow(hessian, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
# plt.show()
# #
# hessian_inv = np.linalg.inv(hessian)
#
# delta_x = hessian_inv@np.expand_dims(grad.flatten(), 0).T
# delta_x = np.reshape(delta_x,shape)

"""
Precondtioned by Hessian
"""
# import glob, os, sys
# vel = np.load("/mnt/others/DATA/Inversion/RNN_Hessian/velocity/Ori_20_20_21x51_part.npy")
# hessian = np.load("/mnt/others/DATA/Inversion/RNN_Hessian/hessian/hessian_0hot.npy")
# grads = [np.load(path).copy() for path in sorted(glob.glob(os.path.join("/mnt/others/DATA/Inversion/RNN_Hessian/grad", "*")))]

# for i in range(5):
#     hessian = np.load("/mnt/others/DATA/Inversion/RNN_Hessian/hessian/hessian_%dhot.npy"%(i))
#     grad = grads[i]-np.sum(grads[0:i])
#     vmin, vmax = np.percentile(grad, [2, 98])
#     plt.imshow(grad, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
#     plt.show()
#
#     shape = grad.shape
#     precon_grad = np.linalg.inv(hessian) @ np.expand_dims(grad.flatten(), 0).T
#     precon_grad = precon_grad.reshape(shape)
#     np.save("/mnt/others/DATA/Inversion/RNN_Hessian/precond/precond_%dshot.npy"%(i), precon_grad)
#
# exit()
#
# hessian = np.load("/mnt/others/DATA/Inversion/RNN_Hessian/hessian/hessian_0hot.npy")
# grads = [np.load(path).copy() for path in sorted(glob.glob(os.path.join("/mnt/others/DATA/Inversion/RNN_Hessian/grad", "*")))]
# grad = np.load("/mnt/others/DATA/Inversion/RNN_Hessian/grad/grad_4hot.npy")
# precon_grad = np.load("/mnt/others/DATA/Inversion/RNN_Hessian/precond/precond_4shot.npy")
# vel = np.load("/mnt/others/DATA/Inversion/RNN_Hessian/velocity/Ori_20_20_21x51_part.npy")
# fig, axes = plt.subplots(1, 3)
# vmin,vmax=np.percentile(grad, [2, 98])
# axes[0].imshow(grad, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
# vmin,vmax=np.percentile(precon_grad, [2, 98])
# axes[1].imshow(precon_grad, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
# vmin,vmax=np.percentile(vel, [2, 98])
# axes[2].imshow(vel, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
# plt.show()
#
# vmin,vmax=np.percentile(hessian_inv, [1, 99])
# plt.imshow(hessian_inv, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
# plt.show()


# Point model
# true = np.ones((51, 101), dtype=np.float32)*1500.
# true[7:,:] = 2000.
#
# # Scatter1
# true[14,31] = 2500
#
# # Scatter2
#
# # Scatter3
#
# init = np.ones_like(true) * 1500.
# init[7:,:] = 2000
# fig, axes = plt.subplots(1, 2)
# vmin,vmax=np.percentile(true, [2, 98])
# axes[0].imshow(true, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
# vmin,vmax=np.percentile(init, [2, 98])
# axes[1].imshow(init, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic, aspect='auto')
#
# plt.show()
# #
# np.save("/mnt/others/DATA/Inversion/RNN_Hessian/velocity/true21x21.npy", true)
# np.save("/mnt/others/DATA/Inversion/RNN_Hessian/velocity/init21x21.npy", init)










































































