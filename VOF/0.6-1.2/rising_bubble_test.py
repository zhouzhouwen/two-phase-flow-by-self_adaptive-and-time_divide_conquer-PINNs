import pandas as pd
from torch.autograd import Variable
import torch
import torch.nn as nn
import seaborn as sns
import time
import scipy.io
import os
import csv
import random
from torch.cuda.amp import autocast, GradScaler
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

class ResidualBlock(nn.Module):
    def __init__(self, NN):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(NN, NN)
        self.linear2 = nn.Linear(NN, NN)
        self.activation = nn.Softsign()  # 或者使用其他激活函数

    def forward(self, x):
        identity = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out += identity  # 添加残差连接
        out = self.activation(out)
        return out

class Net(nn.Module):
    def __init__(self, NL, NN):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(3, NN)
        self.hidden_layers = nn.ModuleList([ResidualBlock(NN) for _ in range(NL)])
        self.output_layer = nn.Linear(NN, 4)
        self.activation = nn.Softsign()  # 选择一个激活函数

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

def reshape_prediction(x, y, t, u):

    ''' Function reshaping the predictions into 2D arrays '''

    return u.reshape(len(t), len(y), len(x), order="C")

# 相对误差
def mean_relative_error(y_pred, y_true,):
    assert y_pred.shape == y_true.shape, "两个矩阵的形状必须相同"

    relative_error = (y_pred - y_true) / y_true
    # relative_error = ((np.abs(y_pred) - np.abs(y_true)) / y_true)
    mean_relative_error = np.mean(np.abs(relative_error))
    # mean_relative_error = np.mean(relative_error)
    return mean_relative_error

# 相对误差
def relative_error(y_pred, y_true,):
    assert y_pred.shape == y_true.shape, "两个矩阵的形状必须相同"
    relative_error = ((y_pred - y_true) / y_true)
    # relative_error = ((np.abs(y_pred) - np.abs(y_true)) / y_true)
    relative_error = process_error(relative_error, y_pred)

    return relative_error

# 如果error里面的某个值是np.inf，说明这个误差值对应的真实值是0，而预测值不是，那么这个时候做一个判断，如果真实值大于1e-5次方，那么将这个误差值改成1，否则将这个误差值改成0
def process_error(error, pre):
    error = np.where(np.isnan(error), 0, error)
    error = np.where(np.isinf(error) & (np.abs(pre) > 1e-2), 0.5, error)
    error = np.where(np.isinf(error), 0, error)
    # error = np.clip(error, -100, 100)
    return error

def l2_relative_error(y_pred, y_true):
    assert y_pred.shape == y_true.shape, "两个矩阵的形状必须相同"
    numerator = np.sqrt(np.sum((y_pred - y_true) ** 2))
    denominator = np.sqrt(np.sum(y_true ** 2))
    # print(numerator)
    # print(denominator)

    l2_relative_error = numerator / denominator
    # 计算 L2 范数的相对误差
    l2_relative_error = np.linalg.norm(y_pred - y_true, 'fro') / np.linalg.norm(y_true, 'fro')

    return l2_relative_error

def test(a,b):
    # 计算矩阵的差值 c = a - b
    c = a - b

    # 计算 L2 范数（欧几里得范数）
    L2_c = np.linalg.norm(c, 'fro')
    L2_a = np.linalg.norm(a, 'fro')

    # 计算相对误差
    relative_error = L2_c / L2_a
    return relative_error

def l2_relative_error_3D(A, B):
    assert A.shape == B.shape, "两个三维矩阵的形状必须相同"
    m, n, p = A.shape
    errors = np.zeros(m)
    for i in range(m):
        errors[i] = l2_relative_error(A[i, :, :], B[i, :, :])
    return errors


device = torch.device("cpu")  # 使用cpu预测
path = "/home/user/ZHOU-Wen/PINN/twophasePINN-master/cfd_data/rising_bubble.h5"
temporal_step_size = 1
spatial_step_size = 1
min_value = 5e-3
interface_number = 7.5
max_a = 1.0
min_a = 0.0
model_file_path = '/home/user/ZHOU-Wen/PINN/two-phase-flow/VOF/0.6-1.2/best_net_pde.pth'
next_data_ic_file_path = '/home/user/ZHOU-Wen/PINN/two-phase-flow/VOF/0.6-1.2/next_data_ic.npy'

# TIME SNAPSHOT SELECTION

indices = np.arange(151)
# indices = np.sort(np.concatenate([indices[0:30:15], indices[30:100:10], indices[100::10], indices[1:4]], axis=0))
indices = np.sort(np.concatenate([indices[31:61:temporal_step_size],], axis=0))
# indices = np.sort(np.concatenate([indices[60:151:1],], axis=0))

with h5py.File(path, "r") as data:
    X = np.array(data["X"])[::spatial_step_size]
    Y = np.array(data["Y"])[::spatial_step_size]
    # time = np.array(data["time"])[start_index:end_index:temporal_step_size]
    time = np.array(data["time"])
    time = time[indices]
    print('time:', time)
    print('time:', time.shape)

    real_a = np.array(data["levelset"])[:, ::spatial_step_size, ::spatial_step_size]
    real_a = real_a[indices]

    real_p = np.array(data["pressure"])[:, ::spatial_step_size, ::spatial_step_size]
    real_p = real_p[indices]

    real_u = np.array(data["velocityX"])[:, ::spatial_step_size, ::spatial_step_size]
    real_u = real_u[indices]

    # real_v = np.array(data["velocityY"])[start_index:end_index:temporal_step_size, ::spatial_step_size, ::spatial_step_size]
    real_v = np.array(data["velocityY"])[:, ::spatial_step_size, ::spatial_step_size]
    real_v = real_v[indices]

    # 创建一个3维矩阵（示例）
    matrix = real_a

    # 遍历并替换元素
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                # if matrix[i][j][k] >= 0:
                #     matrix[i][j][k] = 1.0
                # else:
                #     matrix[i][j][k] = 0.0
                #  注意顺序，如果你把变成0放到上面，下面又被覆盖了
                if -interface_number <= matrix[i][j][k] < interface_number:
                    matrix[i][j][k] = 1 / 2 * (1 + matrix[i][j][k] / interface_number + math.sin(
                        math.pi * matrix[i][j][k] / interface_number) / math.pi)

                if matrix[i][j][k] < -interface_number:
                    matrix[i][j][k] = 0.0

                if matrix[i][j][k] > interface_number:
                    matrix[i][j][k] = 1.0

    real_a = matrix.astype(float)
    real_p = real_p / 1000

# print(np.max(real_a))
# print(np.min(real_a))
# real_a = real_a.ravel()
# non_zero_arr = real_a[real_a != 0]
# temp =np.sort(non_zero_arr)
# print(np.min(temp))

# 非常重要，顺序不能错，因为要保证x，y，t的顺序，所以下面是y,t,x
pt_y, pt_t, pt_x = np.meshgrid(Y, time, X)

pt_x = pt_x.reshape(-1,1)
pt_y = pt_y.reshape(-1,1)
pt_t = pt_t.reshape(-1,1)
# pt_t = pt_t - 1.2
pt_x = Variable(torch.from_numpy(pt_x).float(), requires_grad=True).to(device)
pt_y = Variable(torch.from_numpy(pt_y).float(), requires_grad=True).to(device)
pt_t = Variable(torch.from_numpy(pt_t).float(), requires_grad=True).to(device)

net = torch.load(model_file_path).to(device)
# net = torch.load('/home/user/ZHOU-Wen/PINN/two-phase-flow/VOF/0.6-1.2/best_net_resample_pde.pth')
# net = torch.load('/home/user/ZHOU-Wen/PINN/two-phase-flow/VOF/0.6-1.2/best_net(batchsize).pth')
predict = net(torch.cat([pt_x, pt_y, pt_t], 1)).to(device)
pre_u, pre_v, pre_p, pre_a = predict[:, 0].cpu().data.numpy(), predict[:, 1].cpu().data.numpy(), predict[:, 2].cpu().data.numpy(), predict[:, 3].cpu().data.numpy()


pre_u = reshape_prediction(X, Y, time, pre_u)
pre_v = reshape_prediction(X, Y, time, pre_v)
pre_p = reshape_prediction(X, Y, time, pre_p)
pre_a = reshape_prediction(X, Y, time, pre_a)


# one second
data = pre_a[-1,:,:]
data_array = np.array(data)
# print(data)
# print(data.shape)
cmap = 'jet'
data = np.flipud(data)
plt.imshow(data, cmap=cmap)
plt.colorbar()
plt.title("Result")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


real_u = np.where(np.abs(real_u) < min_value, 0, real_u)
real_v = np.where(np.abs(real_v) < min_value, 0, real_v)
real_a = np.where(np.abs(real_a) < min_value, 0, real_a)

pre_u = np.where(np.abs(pre_u) < min_value, 0, pre_u)
pre_v = np.where(np.abs(pre_v) < min_value, 0, pre_v)
pre_a = np.where(np.abs(pre_a) < min_value, 0, pre_a)

# 使用 np.where 处理pre_a大于1的情况
pre_a = np.where(pre_a > max_a, max_a, pre_a)
# 再次使用 np.where 处理pre_a小于0的情况
pre_a = np.where(pre_a < min_a, min_a, pre_a)

# chuli u v?

def minmax_normalization(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)

def standardization(array):
    mean_val = np.mean(array)
    std_val = np.std(array)
    return (array - mean_val) / std_val

# real_u = standardization(real_u)
# real_v = standardization(real_v)


print(np.unique(real_u))
print(np.unique(real_v))
print(np.unique(real_a))
print(np.unique(pre_u))
print(np.unique(pre_v))
print(np.unique(pre_a))

# convert matrix in form of 2d
for i in range(real_a.shape[0]):
    real_u[i] = np.flipud(real_u[i])
    real_v[i] = np.flipud(real_v[i])
    real_p[i] = np.flipud(real_p[i])
    real_a[i] = np.flipud(real_a[i])
    pre_u[i] = np.flipud(pre_u[i])
    pre_v[i] = np.flipud(pre_v[i])
    pre_p[i] = np.flipud(pre_p[i])
    pre_a[i] = np.flipud(pre_a[i])


def correct_sign_3D(matrix):
    # 获取三维矩阵的维度
    t, m, n = matrix.shape
    # 创建一个与输入矩阵形状相同的标记矩阵
    change_sign_matrix = np.zeros((t, m, n), dtype=bool)

    # 遍历每一个时间切片
    for ti in range(t):
        # 遍历每一个二维切片，注意从1开始到m-1和n-1，以排除最外层
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                # 获取中心元素
                center = matrix[ti, i, j]
                # 获取周围8个元素
                neighbors = matrix[ti, i - 1:i + 2, j - 1:j + 2].flatten()
                # 去掉中心元素
                neighbors = np.delete(neighbors, 4)

                # 判断中心元素与周围元素是否符号一致
                same_sign = np.sign(center) == np.sign(neighbors)

                # 如果中心元素与所有周围元素的符号都不一致，则标记需要改变符号
                if np.all(same_sign == False):
                    change_sign_matrix[ti, i, j] = True

    # 根据标记矩阵，更改需要更改符号的元素
    matrix[change_sign_matrix] *= -1

# correct_sign_3D(pre_u)
# correct_sign_3D(pre_v)


error_u = relative_error(pre_u, real_u)
error_v = relative_error(pre_v, real_v)
error_p = relative_error(pre_p, real_p)
error_a = relative_error(pre_a, real_a)
# print(error_u.shape)

l2_error_u = l2_relative_error_3D(pre_u, real_u)
l2_error_v = l2_relative_error_3D(pre_v, real_v)
l2_error_p = l2_relative_error_3D(pre_p, real_p)
l2_error_a = l2_relative_error_3D(pre_a, real_a)

print('L1 error u:',np.mean(error_u, axis=(1, 2)))
print('L2 error u:', l2_error_u)
print('L1 error v:',np.mean(error_v, axis=(1, 2)))
print('L2 error v:', l2_error_v)
print('L1 error p:',np.mean(error_p, axis=(1, 2)))
print('L2 error p:', l2_error_p)
print('L1 error a:',np.mean(error_a, axis=(1, 2)))
print('L2 error a:', l2_error_a)

# print('mean L1 error u:',np.mean(np.abs(error_u)))
# print('mean L1 error u:',np.mean(np.abs(np.mean(error_u, axis=(1, 2)))))
# print('mean L2 error u:',np.mean(l2_error_u))
# print('mean L1 error v:',np.mean(np.abs(error_v)))
# print('mean L1 error v:',np.mean(np.abs(np.mean(error_v, axis=(1, 2)))))
# print('mean L2 error v:',np.mean(l2_error_v))
# print('mean L1 error p:',np.mean(np.abs(error_p)))
# print('mean L1 error p:',np.mean(np.abs(np.mean(error_p, axis=(1, 2)))))
# print('mean L2 error p:',np.mean(l2_error_p))
# print('mean L1 error a:',np.mean(np.abs(error_a)))
# print('mean L1 error a:',np.mean(np.abs(np.mean(error_a, axis=(1, 2)))))
# print('mean L2 error a:',np.mean(l2_error_a))

# all movie
matrices = [real_u, real_v, real_p, real_a, pre_u, pre_v, pre_p, pre_a, error_u, error_v, error_p, error_a]
# 创建四个子图的布局
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, a11, ax12)) = plt.subplots(3, 4, figsize=(12, 8))
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, a11, ax12]
# 设置子图之间的间距
fig.subplots_adjust(wspace=0.000001, hspace=0.4)

# 设置每个子图的标题
titles = ['CFD benchmark u', 'CFD benchmark v', 'CFD benchmark p', 'CFD benchmark a', 'Improved PINNs u', 'Improved PINNs v', 'Improved PINNs p', 'Improved PINNs a', 'L1 error of u', 'L1 error of v', 'L1 error of p', 'L1 error of a']
for ax, title in zip(axes, titles):
    ax.set_title(title)


# 找到每一对图的全局最大值和最小值
global_min_max_pairs = [
    (np.min(real_u), np.max(real_u)),
    (np.min(real_v), np.max(real_v)),
    (np.min(real_p), np.max(real_p)),
    (np.min(real_a), np.max(real_a))
]

# 设置每个子图的初始图像和colorbar
imgs = []
for i, (ax, matrix) in enumerate(zip(axes, matrices)):
    if i < 4:  # 图1到图4
        vmin, vmax = global_min_max_pairs[i]
        im = ax.imshow(matrix[0], animated=True, cmap='rainbow', vmin=vmin, vmax=vmax)
    elif 4 <= i < 8:  # 图5到图8
        vmin, vmax = global_min_max_pairs[i - 4]
        im = ax.imshow(matrix[0], animated=True, cmap='rainbow', vmin=vmin, vmax=vmax)
    else:  # 图9到图12
        im = ax.imshow(matrix[0], animated=True, cmap='rainbow', vmin=-3, vmax=3)

    fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04)  # 调整 colorbar 的长度
    imgs.append(im)

# 设置所有子图的坐标轴刻度
for ax in axes:
    ax.set_xticks(np.linspace(0, 256/spatial_step_size, 3))  # x轴刻度设置为-0.5, 0, 0.5
    ax.set_yticks(np.linspace(0, 512/spatial_step_size, 3))  # y轴刻度设置为-1, 0, 1
    ax.set_xticklabels([-0.5, 0, 0.5])  # 设置x轴刻度标签
    ax.set_yticklabels([1, 0, -1])  # 设置y轴刻度标签

def updatefig(i):
    # 每一帧更新图像的内容
    for img, matrix in zip(imgs, matrices):
        img.set_array(matrix[i])
    return imgs

ani = animation.FuncAnimation(fig, updatefig, frames=real_a.shape[0], interval=100, blit=True, repeat=True)
ani.save('uvpa_results.mp4', writer='ffmpeg', fps=10, dpi=300)
plt.show()

# Generate initial conditions for next time.
y_ic, t_ic, x_ic = np.meshgrid(Y, time[-1], X)

x_ic = x_ic.reshape(-1,1)
y_ic = y_ic.reshape(-1,1)
t_ic = t_ic.reshape(-1,1)
# pt_t = pt_t - 1.2
x_ic = Variable(torch.from_numpy(x_ic).float(), requires_grad=True).to(device)
y_ic = Variable(torch.from_numpy(y_ic).float(), requires_grad=True).to(device)
t_ic = Variable(torch.from_numpy(t_ic).float(), requires_grad=True).to(device)

predict_ic = net(torch.cat([x_ic, y_ic, t_ic], 1)).to(device)
pre_u_ic, pre_v_ic, pre_p_ic, pre_a_ic = predict_ic[:, 0].cpu().data.numpy(), predict_ic[:, 1].cpu().data.numpy(), predict_ic[:, 2].cpu().data.numpy(), predict_ic[:, 3].cpu().data.numpy()

# 将x_ic, y_ic, t_ic从CUDA转移到CPU
x_ic = x_ic.cpu().data.numpy()
y_ic = y_ic.cpu().data.numpy()
t_ic = t_ic.cpu().data.numpy()

pre_u_ic = pre_u_ic.reshape(-1,1)
pre_v_ic = pre_v_ic.reshape(-1,1)
pre_p_ic = pre_p_ic.reshape(-1,1)
pre_a_ic = pre_a_ic.reshape(-1,1)

# 合并所有列到一个单一数组
data_combined = np.hstack([x_ic, y_ic, t_ic, pre_u_ic, pre_v_ic, pre_p_ic, pre_a_ic])

# 保存数据为.npy文件
np.save(next_data_ic_file_path, data_combined)



# 画出界面显示比较
# Initialize plot
fig, ax = plt.subplots()

def animate(i):
    ax.clear()  # Clear the plot to draw the new frame
    # Plot real_a values between 0.45 and 0.55 with black solid lines
    real_mask = (real_a[i] > 0.05) & (real_a[i] < 0.95)
    ax.imshow(real_mask, cmap='gray', alpha=0.5)  # Use alpha for transparency
    ax.contour(real_mask, colors='black',)

    # Plot pre_a values between 0.45 and 0.55 with red dashed lines
    pre_mask = (pre_a[i] > 0.05) & (pre_a[i] < 0.95)
    ax.contour(pre_mask, colors='r', linestyles='dashed')
    # Set the title to indicate the frame number
    ax.set_title(f"Frame {i+1}")
    ax.set_xticks(np.linspace(0, 256/spatial_step_size, 3))  # x轴刻度设置为-0.5, 0, 0.5
    ax.set_yticks(np.linspace(0, 512/spatial_step_size, 3))  # y轴刻度设置为-1, 0, 1
    ax.set_xticklabels([-0.5, 0, 0.5])  # 设置x轴刻度标签
    ax.set_yticklabels([1, 0, -1])  # 设置y轴刻度标签

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=real_a.shape[0], interval=200)
ani.save('interface_results.mp4', writer='ffmpeg', fps=10, dpi=300)
plt.show()



# 画出质心比较
# Defining the y and x coordinates as specified
y_coords = np.linspace(2, 0, int(512/spatial_step_size))
x_coords = np.linspace(1, 0, int(256/spatial_step_size))

# x_coords = np.linspace(0.49414065+0.5, -0.49804688+0.5, int(256/spatial_step_size))
# y_coords = np.linspace(0.9941406+1, -0.9980469+1, int(512/spatial_step_size))

# Initialize lists to hold the centroid values for each time slice
center_mass_values_real = []
center_mass_values_pre = []

# Define a function to calculate the centroid of a 2D slice
def calculate_center_mass(matrix_slice, x_coords, y_coords):
    integral_num = np.sum((1 - matrix_slice) * y_coords[:, None], axis=(0, 1))
    integral_den = np.sum(1 - matrix_slice, axis=(0, 1))
    center_mass = integral_num / integral_den if integral_den != 0 else 0
    return center_mass

# Loop through each time slice and calculate the center_mass for real_a and pre_a
for t in range(real_a.shape[0]):
    real_slice = real_a[t, :, :]
    pre_slice = pre_a[t, :, :]
    center_mass_values_real.append(calculate_center_mass(real_slice, x_coords, y_coords))
    center_mass_values_pre.append(calculate_center_mass(pre_slice, x_coords, y_coords))

L1_error_center_mass = [(pre - real) / real for pre, real in zip(center_mass_values_pre, center_mass_values_real)]
print("center_mass_values_real:", np.array(center_mass_values_real))
print("center_mass_values_pre:", np.array(center_mass_values_pre))
print("L1 error center_mass:", np.array(L1_error_center_mass))

# Plotting the centroid values over time
time_points = np.arange(1, real_a.shape[0]+1)
plt.figure(figsize=(12, 6))
plt.plot(time_points, center_mass_values_real, label='Center of mass by benchmark')
plt.plot(time_points, center_mass_values_pre, label='Center of mass by improved PINNs')
plt.xlabel('Time')
plt.title('Center of mass over time')
plt.legend()
plt.show()



# 画出上升速度比较
# Initialize lists to hold the vc values for each time slice
vc_values_real = []
vc_values_pre = []

# Define a function to calculate vc for a 2D slice
def calculate_vc(matrix_slice, velocity_slice, x_coords, y_coords):
    integral_num = np.sum(velocity_slice * (1 - matrix_slice) * y_coords[:, None], axis=(0, 1))
    integral_den = np.sum(1 - matrix_slice, axis=(0, 1))
    vc = integral_num / integral_den if integral_den != 0 else 0
    return vc

# Loop through each time slice and calculate vc for real_a and pre_a using real_v and pre_v
for t in range(real_a.shape[0]):
    real_slice = real_a[t, :, :]
    pre_slice = pre_a[t, :, :]
    real_velocity_slice = real_v[t, :, :]
    pre_velocity_slice = pre_v[t, :, :]
    vc_values_real.append(calculate_vc(real_slice, real_velocity_slice, x_coords, y_coords))
    vc_values_pre.append(calculate_vc(pre_slice, pre_velocity_slice, x_coords, y_coords))

L1_error_vc = [(pre - real) / real for pre, real in zip(vc_values_pre, vc_values_real)]
print("vc_values_real:", np.array(vc_values_real))
print("vc_values_pre:", np.array(vc_values_pre))
print("L1_error_vc:", np.array(L1_error_vc))

# Plotting the vc values over time
time_points = np.arange(1, real_a.shape[0]+1)
plt.figure(figsize=(12, 6))
plt.plot(time_points, vc_values_real, label='Vc by benchmark')
plt.plot(time_points, vc_values_pre, label='Vc by improved PINNs')
plt.xlabel('Time')
plt.title('Rising velocity over time')
plt.legend()
plt.show()
