import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import h5py
path = "/home/user/ZHOU-Wen/PINN/twophasePINN-master/cfd_data/rising_bubble.h5"
temporal_step_size = 2
spatial_step_size = 2
min_value = 1e-3
# mesh_size = 0.003906
interface_number = 7.5
number_of_pde = 400

with h5py.File(path, "r") as data:
    X = np.array(data["X"])
    Y = np.array(data["Y"])
    times = np.array(data["time"])
    levelset = np.array(data["levelset"])
    velocityX = np.array(data["velocityX"])
    velocityY = np.array(data["velocityY"])
    pressure = np.array(data["pressure"])
    # for key in data.keys():
    #     print(key)

# TIME SNAPSHOT SELECTION
indices = np.arange(len(times))
# indices = np.sort(np.concatenate([indices[0:30:15], indices[30:100:10], indices[100::10], indices[1:4]], axis=0))
# indices = np.sort(np.concatenate([indices[1:3],indices[0:70:5],], axis=0))
indices = np.sort(np.concatenate([indices[31:61:temporal_step_size],], axis=0))
times = times[indices]
levelset = levelset[indices]
velocityX = velocityX[indices]
velocityY = velocityY[indices]
pressure = pressure[indices]

# 创建一个3维矩阵（示例）
matrix = levelset
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
                # https://en.wikipedia.org/wiki/Level-set_method ; Conversion of VOF and LS methods
                matrix[i][j][k] = 1/2 * (1 + matrix[i][j][k]/interface_number + math.sin(math.pi*matrix[i][j][k]/interface_number) / math.pi)

            if matrix[i][j][k] < -interface_number:
                matrix[i][j][k] = 0.0

            if matrix[i][j][k] > interface_number:
                matrix[i][j][k] = 1.0


data_A = matrix.astype(float)
pressure = pressure / 1000

velocityX = np.where(np.abs(velocityX) < min_value, 0, velocityX)
velocityY = np.where(np.abs(velocityY) < min_value, 0, velocityY)
data_A = np.where(data_A < min_value, 0, data_A)


print(np.unique(velocityX))
print(np.unique(velocityY))
# print(np.unique(pressure))
# print(np.unique(data_A))


def minmax_normalization(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)

def standardization(array):
    mean_val = np.mean(array)
    std_val = np.std(array)
    return (array - mean_val) / std_val


# velocityX = standardization(velocityX)
# velocityY = standardization(velocityY)



print('----------information----------')
print('Range of velocityX:',np.unique(velocityX))
print('Range of velocityY:',np.unique(velocityY))
print('Range of pressure:',np.unique(pressure))
print('Range of data_A:',np.unique(data_A))
print("Time slice:", times)
print("Number of time slice: ", len(times))
print("Start time:",times[0])
t_bounds = [times[0], times[-1]]
x_bounds = [X[0], X[-1]]
y_bounds = [Y[0], Y[-1]]

# IC points
interval_ic = spatial_step_size
x = X[0::interval_ic]  # odd_position_elements
y = Y[0::interval_ic]  # odd_position_elements
# print(x)
# print(y)
# print(x.shape)
# print(y.shape)
x, y = np.meshgrid(x, y)
x_ic = x.ravel().reshape(-1,1)
y_ic = y.ravel().reshape(-1,1)
t_ic = np.ones_like(x_ic)*times[0]
u_ic = velocityX[0, ::interval_ic, ::interval_ic].ravel().reshape(-1,1)
v_ic = velocityY[0, ::interval_ic, ::interval_ic].ravel().reshape(-1,1)
p_ic = pressure[0, ::interval_ic, ::interval_ic].ravel().reshape(-1,1)
a_ic = data_A[0, ::interval_ic, ::interval_ic].ravel().reshape(-1,1)


plt.figure(figsize=(4,8))
plt.scatter(x_ic,y_ic,s=1)
plt.show()
data_ic = np.concatenate([x_ic, y_ic, t_ic, u_ic, v_ic, p_ic, a_ic], 1)
print(data_ic.shape)
# np.save('/home/user/ZHOU-Wen/PINN/two-phase-flow/VOF/0.6-1.2/data_ic.npy', data_ic)



# BC points, counter-clock, bottom is first
interval_bc = spatial_step_size
x = X[0::interval_bc]  # odd_position_elements
y = Y[0::interval_bc]  # odd_position_elements

def extract_layer(x, y, layer):
    if layer == 1:
        x_bc = np.hstack([x, np.tile(x[-1], len(y) - 2), x[::-1], np.tile(x[0], len(y) - 2)])
        y_bc = np.hstack([np.tile(y[0], len(x)), y[1:-1], np.tile(y[-1], len(x)), y[::-1][1:-1]])
        return x_bc, y_bc
    elif layer >= 2:
        m = len(x)
        n = len(y)
        x_part = x[layer - 1:m - layer + 1]
        y_part = y[layer - 1:n - layer + 1]
        x_bc = np.hstack(
            [x_part, np.tile(x_part[-1], len(y_part) - 2), x_part[::-1], np.tile(x_part[0], len(y_part) - 2)])
        y_bc = np.hstack(
            [np.tile(y_part[0], len(x_part)), y_part[1:-1], np.tile(y_part[-1], len(x_part)), y_part[::-1][1:-1]])
        return x_bc, y_bc

def extract_nth_shell_2d(matrix, n):
    rows, cols = matrix.shape
    if n < 1 or 2 * n > min(rows, cols) + 1:
        return []
    nth_shell = []
    # 下边
    nth_shell.extend(matrix[n-1, n-1:cols-n+1])
    # 右边
    nth_shell.extend(matrix[n:rows-n, cols-n])
    # 上边
    nth_shell.extend(matrix[rows-n, n-1:cols-n+1][::-1])
    # 左边
    nth_shell.extend(matrix[rows-n-1:n-1:-1, n-1])

    return np.array(nth_shell)

num_1 = 1
# num_2 = 3
# num_3 = 5
# 指定的最外面的第1,3,5层
x_bc_outer_1, y_bc_outer_1 = extract_layer(x, y, num_1)
# print(x_bc_outer_1.shape)
# x_bc_outer_2, y_bc_outer_2 = extract_layer(x, y, num_2)
# x_bc_outer_3, y_bc_outer_3 = extract_layer(x, y, num_3)

# x_bc = np.concatenate([x_bc_outer_1,x_bc_outer_2,x_bc_outer_3])
# y_bc = np.concatenate([y_bc_outer_1,y_bc_outer_2,y_bc_outer_3])
# t_bc = np.repeat(times,x_bc_outer_1.shape[0]+x_bc_outer_2.shape[0]+x_bc_outer_3.shape[0]).reshape(-1,1)

x_bc = np.concatenate([x_bc_outer_1,])
y_bc = np.concatenate([y_bc_outer_1,])
t_bc = np.repeat(times,x_bc_outer_1.shape[0],).reshape(-1,1)


# 指定的最外面的第1层
# x_bc = np.hstack([x, np.tile(x[-1], len(y)-2), x[::-1], np.tile(x[0], len(y)-2)])
# y_bc = np.hstack([np.tile(y[0], len(x)), y[1:-1], np.tile(y[-1], len(x)), y[::-1][1:-1]])
# t_bc = np.repeat(times,x_bc.shape[0]).reshape(-1,1)


x_bc = np.tile(x_bc, len(indices)).reshape(-1,1)
y_bc = np.tile(y_bc, len(indices)).reshape(-1,1)


# 指定的最外面的第1,3,5层
u_bc = np.array([])
for velocityX_snapshot in velocityX:
    velocityX_snapshot = velocityX_snapshot[::interval_bc,::interval_bc]
    u_bc = np.append(u_bc, extract_nth_shell_2d(velocityX_snapshot, num_1))
    # u_bc = np.append(u_bc, extract_nth_shell_2d(velocityX_snapshot, num_2))
    # u_bc = np.append(u_bc, extract_nth_shell_2d(velocityX_snapshot, num_3))

v_bc = np.array([])
for velocityY_snapshot in velocityY:
    velocityY_snapshot = velocityY_snapshot[::interval_bc,::interval_bc]
    v_bc = np.append(v_bc, extract_nth_shell_2d(velocityY_snapshot, num_1))
    # v_bc = np.append(v_bc, extract_nth_shell_2d(velocityY_snapshot, num_2))
    # v_bc = np.append(v_bc, extract_nth_shell_2d(velocityY_snapshot, num_3))

p_bc = np.array([])
for pressure_snapshot in pressure:
    pressure_snapshot = pressure_snapshot[::interval_bc,::interval_bc]
    p_bc = np.append(p_bc, extract_nth_shell_2d(pressure_snapshot, num_1))
    # p_bc = np.append(p_bc, extract_nth_shell_2d(pressure_snapshot, num_2))
    # p_bc = np.append(p_bc, extract_nth_shell_2d(pressure_snapshot, num_3))

a_bc = np.array([])
for data_A_snapshot in data_A:
    data_A_snapshot = data_A_snapshot[::interval_bc,::interval_bc]
    a_bc = np.append(a_bc, extract_nth_shell_2d(data_A_snapshot, num_1))
    # a_bc = np.append(a_bc, extract_nth_shell_2d(data_A_snapshot, num_2))
    # a_bc = np.append(a_bc, extract_nth_shell_2d(data_A_snapshot, num_3))


# 指定的最外面的第1层
# def extract_outer_shell_2d(matrix):
#     outer_shell = []
#     # 下边
#     outer_shell.extend(matrix[0, :])
#     # 右边
#     outer_shell.extend(matrix[1:-1, -1])
#     # 上边
#     if matrix.shape[0] > 1:
#         outer_shell.extend(matrix[-1, :][::-1])
#     # 左边
#     if matrix.shape[0] > 2:
#         outer_shell.extend(matrix[1:-1, 0][::-1])
#
#     return np.array(outer_shell)
#
# u_bc = np.array([])
# for velocityX_snapshot in velocityX:
#     velocityX_snapshot = velocityX_snapshot[::interval_bc,::interval_bc]
#     u_bc = np.append(u_bc, extract_outer_shell_2d(velocityX_snapshot))
#
# v_bc = np.array([])
# for velocityY_snapshot in velocityY:
#     velocityY_snapshot = velocityY_snapshot[::interval_bc,::interval_bc]
#     v_bc = np.append(v_bc, extract_outer_shell_2d(velocityY_snapshot))
#
# p_bc = np.array([])
# for pressure_snapshot in pressure:
#     pressure_snapshot = pressure_snapshot[::interval_bc,::interval_bc]
#     p_bc = np.append(p_bc, extract_outer_shell_2d(pressure_snapshot))
#
# a_bc = np.array([])
# for data_A_snapshot in data_A:
#     data_A_snapshot = data_A_snapshot[::interval_bc,::interval_bc]
#     a_bc = np.append(a_bc, extract_outer_shell_2d(data_A_snapshot))


u_bc = u_bc.reshape(-1,1)
v_bc = v_bc.reshape(-1,1)
p_bc = p_bc.reshape(-1,1)
a_bc = a_bc.reshape(-1,1)

data_bc = np.concatenate([x_bc, y_bc, t_bc, u_bc, v_bc, p_bc, a_bc], 1)

# temp = data_bc[data_bc[:,2] == 1.2000041]
# fig = plt.figure(figsize=(4,8))
# plt.scatter(temp[:,0],temp[:,1],s=1)
# plt.show()

plt.figure(figsize=(4,8))
plt.scatter(x_bc,y_bc,s=1)
plt.show()
print(data_bc.shape)
np.save('/home/user/ZHOU-Wen/PINN/two-phase-flow/VOF/0.6-1.2/data_bc.npy', data_bc)



# PDE points
interval_pde = spatial_step_size
x = X[0::interval_pde]  # odd_position_elements
y = Y[0::interval_pde]  # odd_position_elements
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
x_pde = np.array([])
# print(x.shape)
y_pde = np.array([])
from scipy.stats import qmc


for i in range(len(times)):

    sampler = qmc.Sobol(d=1, scramble=True)
    # sample = sampler.random_base2(m=10)
    sample = sampler.integers(l_bounds=0, u_bounds=int(512/interval_pde*256/interval_pde - 1), n=number_of_pde, endpoint=True)
    sample = sample.ravel()


    random_samples = np.sort(np.random.choice(int(512/interval_pde*256/interval_pde - 1), number_of_pde, replace=False))
    # print(random_samples)

    # uniform_samples = np.linspace(0, int(512/interval_pde*256/interval_pde - 1), number_of_pde, dtype=int)
    # # print(uniform_samples)

    x_pde = np.append(x_pde,x[random_samples])
    y_pde = np.append(y_pde,y[random_samples])

    # fig = plt.figure(figsize=(4,8))
    # plt.scatter(x[sample],y[sample],s=1)
    # plt.show()

x_pde = x_pde.reshape(-1,1)
y_pde = y_pde.reshape(-1,1)
t_pde = np.repeat(times,number_of_pde).reshape(-1,1)
data_pde = np.concatenate([x_pde, y_pde, t_pde], 1)



velocityX = velocityX[:, ::spatial_step_size, ::spatial_step_size]
velocityY = velocityY[:, ::spatial_step_size, ::spatial_step_size]
pressure = pressure[:, ::spatial_step_size, ::spatial_step_size]
data_A = data_A[:, ::spatial_step_size, ::spatial_step_size]

# # convert matrix in form of 2d
# for i in range(velocityX.shape[0]):
#     velocityX[i] = np.flipud(velocityX[i])
#     velocityY[i] = np.flipud(velocityY[i])
#     pressure[i] = np.flipud(pressure[i])
#     data_A[i] = np.flipud(data_A[i])

# x_points 和 y_points 的定义
x_points = np.linspace(-0.49804688, 0.49414065, int(256/spatial_step_size))
y_points = np.linspace(-0.9980469, 0.9941406, int(512/spatial_step_size))

# 按照 t_pde 排序 data_pdes
sorted_data_pdes = data_pde[np.argsort(data_pde[:, 2])]

# 获取 t_pde 的唯一值，并找到对应的索引
unique_t_pdes, unique_t_indices = np.unique(sorted_data_pdes[:, 2], return_inverse=True)

# 初始化一个空的列表用于存储新的数据
extended_data_pdes = []

# 循环遍历 sorted_data_pdes
for i, row in enumerate(sorted_data_pdes):
    x_pde, y_pde, t_pde = row

    # 使用 unique_t_indices 来找到 t_pde 对应的 velocityX 层的索引
    t_index = unique_t_indices[i]

    # 找到 t_index 对应的 velocityX 层
    velocityX_layer = velocityX[t_index]
    velocityY_layer = velocityY[t_index]
    pressure_layer = pressure[t_index]
    data_A_layer = data_A[t_index]

    # 找到 x_pde 和 y_pde 对应的最近的 x_points 和 y_points 索引
    x_index = np.argmin(np.abs(x_points - x_pde))
    y_index = np.argmin(np.abs(y_points - y_pde))
    # print(x_index)
    # print(y_index)
    # 从 velocityX_layer 中获取对应的速度值
    xxx = velocityX_layer[y_index, x_index]
    yyy = velocityY_layer[y_index, x_index]
    ppp = pressure_layer[y_index, x_index]
    aaa = data_A_layer[y_index, x_index]

    # 在当前行后面加上找到的速度值
    new_row = np.append(row, [xxx,yyy,ppp,aaa])

    # 添加到 extended_data_pdes
    extended_data_pdes.append(new_row)

# 转换为 NumPy 数组
extended_data_pdes = np.array(extended_data_pdes)
print(extended_data_pdes.shape)
np.save('/home/user/ZHOU-Wen/PINN/two-phase-flow/VOF/0.6-1.2/extended_data_pdes.npy', extended_data_pdes)
# print(data_pde)
# temp = data_pde[data_pde[:,2] == 1.40009975]
fig = plt.figure(figsize=(4,8))
plt.scatter(extended_data_pdes[number_of_pde*3:number_of_pde*4,0],extended_data_pdes[number_of_pde*3:number_of_pde*4,1],s=1)
plt.show()
# print(data_pde.shape)
# np.save('/home/user/ZHOU-Wen/PINN/two-phase-flow/VOF/0.6-1.2/data_pde.npy', data_pde)
print('saved')