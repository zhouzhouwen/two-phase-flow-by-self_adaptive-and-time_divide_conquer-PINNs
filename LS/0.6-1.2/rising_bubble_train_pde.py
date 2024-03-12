import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import time
import scipy.io
import os
import csv
import random
from torch.cuda.amp import autocast, GradScaler
import math

def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(3)

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

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def NS_bubble(x, y, t, net):
    predict = net(torch.cat([x, y, t], 1))
    # reshape is very important and necessary
    u, v, p, a = predict[:, 0].reshape(-1, 1), predict[:, 1].reshape(-1, 1), predict[:, 2].reshape(-1, 1), predict[:,3].reshape(-1, 1)

    # 使用 clamp 函数进行截断
    a = torch.clamp(a, min=min_a, max=max_a)

    u_x = gradients(u, x)
    u_y = gradients(u, y)
    u_t = gradients(u, t)
    u_xx = gradients(u, x, 2)
    u_yy = gradients(u, y, 2)

    v_x = gradients(v, x)
    v_y = gradients(v, y)
    v_t = gradients(v, t)
    v_xx = gradients(v, x, 2)
    v_yy = gradients(v, y, 2)

    p_x = gradients(p, x)
    p_y = gradients(p, y)

    # Heaviside
    # Define the interface_number
    
    interface_number = 2 * 1.5 * mesh_size  # 2的意思是本来空间的分辨率你已经处以了2
    # Initialize the output tensor with the same shape as a
    output = torch.zeros_like(a)
    # Implement the Heaviside function transformations
    # Values smaller than -interface_number become 0
    output[a < -interface_number] = 0
    # Values larger than interface_number become 1
    output[a > interface_number] = 1
    # Values between -interface_number and interface_number undergo a special transformation
    mask = (a >= -interface_number) & (a <= interface_number)
    output[mask] = 0.5 * (1 + a[mask] / interface_number + torch.sin(a[mask] / interface_number) / math.pi)

    a_x = gradients(a, x)
    a_y = gradients(a, y)
    a_t = gradients(a, t)
    a_xx = gradients(a, x, 2)
    a_yy = gradients(a, y, 2)
    a_xy = gradients(a, y, 2)

    mu = mu2 + (mu1 - mu2) * output
    mu_x = (mu1 - mu2) * gradients(output, x)
    mu_y = (mu1 - mu2) * gradients(output, y)
    rho = rho2 + (rho1 - rho2) * output

    abs_interface_grad = torch.sqrt(torch.square(a_x) + torch.square(a_y) + np.finfo(float).eps)
    # curvature = - (gradients(((a_x + a_y) / abs_interface_grad), x) + gradients(((a_x + a_y) / abs_interface_grad), y))
    # curvature = - (gradients(((a_x+a_y)/(torch.sqrt(torch.square(a_x) + torch.square(a_y) + np.finfo(float).eps))), x)+gradients(((a_x+a_y)/(torch.sqrt(torch.square(a_x) + torch.square(a_y) + np.finfo(float).eps))), y))
    # print(curvature)
    curvature = - ((a_xx + a_yy) / abs_interface_grad - (
                a_x ** 2 * a_xx + a_y ** 2 * a_yy + 2 * a_x * a_y * a_xy) / torch.pow(abs_interface_grad, 3))
    # print(curvature)
    # print('xxxxxx')

    rho_ref = rho2

    one_Re = mu / (rho_ref * U_ref * L_ref)
    one_Re_x = mu_x / (rho_ref * U_ref * L_ref)
    one_Re_y = mu_y / (rho_ref * U_ref * L_ref)
    one_We = sigma / (rho_ref * (U_ref ** 2) * L_ref)
    one_Fr = g * L_ref / (U_ref ** 2)

    # δ(ϕ)
    # 创建一个和 a 形状相同的张量来存储结果
    result = torch.zeros_like(a)
    # 找出哪些元素的绝对值大于 epsilon，并将这些元素设置为 0
    mask_gt_epsilon = torch.abs(a) > interface_number
    result[mask_gt_epsilon] = 0
    # 对于绝对值小于等于 epsilon 的元素，进行特定的计算
    mask_le_epsilon = torch.abs(a) <= interface_number
    result[mask_le_epsilon] = (1 / (2 * interface_number)) * (
                1 + torch.cos(math.pi * a[mask_le_epsilon] / interface_number))

    # PDE_u = (u_t + u * u_x + v * u_y) * rho / rho_ref + p_x - one_We * curvature * a_x - one_Re * (u_xx + u_yy) - 2.0 * one_Re_x * u_x - one_Re_y * (u_y + v_x)
    # PDE_v = (v_t + u * v_x + v * v_y) * rho / rho_ref + p_y - one_We * curvature * a_y - one_Re * (v_xx + v_yy) - rho / rho_ref * one_Fr - 2.0 * one_Re_y * v_y - one_Re_x * (u_y + v_x)
    # f_u = (u_t + u * u_x + v * u_y) * rho / rho_ref + p_x - one_We * curvature * a_x - one_Re * (u_xx + u_yy) - 2.0 * one_Re_x * u_x - one_Re_y * (u_y + v_x)
    f_u = ((u_t + u * u_x + v * u_y) * rho + p_x - one_We * curvature * a_x * result - one_Re * (
                u_xx + u_yy) - 2.0 * one_Re_x * u_x - one_Re_y * (u_y + v_x)) / 100000
    # f_v = (v_t + u * v_x + v * v_y) * rho / rho_ref + p_y - one_We * curvature * a_y - one_Re * (v_xx + v_yy) - rho / rho_ref * one_Fr - 2.0 * one_Re_y * v_y - one_Re_x * (u_y + v_x)
    f_v = ((v_t + u * v_x + v * v_y) * rho + p_y - one_We * curvature * a_y * result - one_Re * (
                v_xx + v_yy) - rho * one_Fr - 2.0 * one_Re_y * v_y - one_Re_x * (u_y + v_x)) / 100000

    f_e = u_x + v_y

    f_a = a_t + u * a_x + v * a_y

    return f_u, f_v, f_e, f_a

def gradients(y, x, order=1):
    if order == 1:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),create_graph=True,only_inputs=True, )[0]
    else:
        return gradients(gradients(y, x), x, order=order - 1)

device = torch.device("cuda")	# 使用gpu训练

loss_path = '/home/user/ZHOU-Wen/PINN/two-phase-flow/LS/0.6-1.2/loss.csv'
if (os.path.exists(loss_path)):
    # 存在，则删除文件
    os.remove(loss_path)
header = ['total_loss',
          'mse_f1','mse_f2','mse_f3','mse_f4',
          'mse_u_ic' ,'mse_v_ic' ,'mse_p_ic','mse_a_ic',
          'mse_u_bc' ,'mse_v_bc','mse_p_bc' ,'mse_a_bc',
          'mse_u_pde' ,'mse_v_pde','mse_p_pde' ,'mse_a_pde',
          'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
with open(loss_path, 'a', encoding='utf-8', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(header)

mu1 = 1
mu2 = 10
sigma = 24.5
g = -0.98
rho1 = 100
rho2 = 1000
# U_ref = (2*0.98*0.25)**0.5
U_ref = 1.0
# L_ref = 0.25*2
L_ref = 0.25
max_a = 1.0
min_a = -1.0
mesh_size = 0.00391

# BC
# x_bc指的是流域的四条边，其他同理
# PS: y is the edge of x
data_bc = np.load('/home/user/ZHOU-Wen/PINN/two-phase-flow/LS/0.6-1.2/data_bc.npy')
# np.random.shuffle(data_bc)
x_bc = Variable(torch.from_numpy(data_bc[:, 0].reshape(-1, 1)).float(), requires_grad=False).to(device)
y_bc = Variable(torch.from_numpy(data_bc[:, 1].reshape(-1, 1)).float(), requires_grad=False).to(device)
t_bc = Variable(torch.from_numpy(data_bc[:, 2].reshape(-1, 1)).float(), requires_grad=False).to(device)
u_bc = Variable(torch.from_numpy(data_bc[:, 3].reshape(-1, 1)).float(), requires_grad=False).to(device)
v_bc = Variable(torch.from_numpy(data_bc[:, 4].reshape(-1, 1)).float(), requires_grad=False).to(device)
p_bc = Variable(torch.from_numpy(data_bc[:, 5].reshape(-1, 1)).float(), requires_grad=False).to(device)
a_bc = Variable(torch.from_numpy(data_bc[:, 6].reshape(-1, 1)).float(), requires_grad=False).to(device)

# IC
data_ic = np.load('/home/user/ZHOU-Wen/PINN/two-phase-flow/LS/0-0.6/next_data_ic.npy')
# np.random.shuffle(data_ic)
x_ic = Variable(torch.from_numpy(data_ic[:, 0].reshape(-1, 1)).float(), requires_grad=False).to(device)
y_ic = Variable(torch.from_numpy(data_ic[:, 1].reshape(-1, 1)).float(), requires_grad=False).to(device)
t_ic = Variable(torch.from_numpy(data_ic[:, 2].reshape(-1, 1)).float(), requires_grad=False).to(device)
u_ic = Variable(torch.from_numpy(data_ic[:, 3].reshape(-1, 1)).float(), requires_grad=False).to(device)
v_ic = Variable(torch.from_numpy(data_ic[:, 4].reshape(-1, 1)).float(), requires_grad=False).to(device)
p_ic = Variable(torch.from_numpy(data_ic[:, 5].reshape(-1, 1)).float(), requires_grad=False).to(device)
a_ic = Variable(torch.from_numpy(data_ic[:, 6].reshape(-1, 1)).float(), requires_grad=False).to(device)

# PDE
# data_pde = np.load('/home/user/ZHOU-Wen/PINN/two-phase-flow/LS/0.6-1.2/data_pde.npy')
# np.random.shuffle(data_pde)
data_pde = np.load('/home/user/ZHOU-Wen/PINN/two-phase-flow/LS/0.6-1.2/extended_data_pdes.npy')
# data_pde = np.load('/home/user/ZHOU-Wen/PINN/two-phase-flow/LS/0.6-1.2/extended_resample_data_pdes.npy')
x_pde = Variable(torch.from_numpy(data_pde[:, 0].reshape(-1, 1)).float(), requires_grad=True).to(device)
y_pde = Variable(torch.from_numpy(data_pde[:, 1].reshape(-1, 1)).float(), requires_grad=True).to(device)
t_pde = Variable(torch.from_numpy(data_pde[:, 2].reshape(-1, 1)).float(), requires_grad=True).to(device)

u_pde = Variable(torch.from_numpy(data_pde[:, 3].reshape(-1, 1)).float(), requires_grad=False).to(device)
v_pde = Variable(torch.from_numpy(data_pde[:, 4].reshape(-1, 1)).float(), requires_grad=False).to(device)
p_pde = Variable(torch.from_numpy(data_pde[:, 5].reshape(-1, 1)).float(), requires_grad=False).to(device)
a_pde = Variable(torch.from_numpy(data_pde[:, 6].reshape(-1, 1)).float(), requires_grad=False).to(device)

log_var_a = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_b = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_c = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_d = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_e = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_f = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_g = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_h = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_i = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_j = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_k = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_l = Variable(1 * torch.ones(1).cuda(), requires_grad=True)

log_var_m = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_n = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_o = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_p = Variable(1 * torch.ones(1).cuda(), requires_grad=True)


net = Net(10, 100)
net.apply(weights_init).to(device)

# get all parameters (model parameters + task dependent log variances)

params = ([p for p in net.parameters()] + [log_var_a] + [log_var_b] + [log_var_c] + [log_var_d] + [log_var_e] + [
    log_var_f] + [log_var_g] + [log_var_h] + [log_var_i] + [log_var_j] + [log_var_k] + [log_var_l] + [log_var_m] + [log_var_n] + [log_var_o] + [log_var_p])

mse_cost_function = torch.nn.MSELoss(reduction='mean') # Mean squared error
# mse_cost_function_cross = torch.nn.BCELoss(reduction='mean') # Mean squared error
# optimizer = torch.optim.LBFGS(net.parameters())
# optimizer = torch.optim.Adam(params)
optimizer = torch.optim.AdamW(params, lr=1e-3)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=300, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08)

# 定义一个lambda函数来实现自定义的学习率调整策略
def lr_lambda(epoch):
    if epoch < 30000:
        return 1.0
    elif epoch < 60000:
        return 0.1
    elif epoch < 80000:
        return 0.01
    elif epoch < 90000:
        return 0.001
    else:
        return 0.0001

# 使用LambdaLR和刚刚定义的lambda函数创建一个调度器
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, verbose=False)

# define MSE loss criterion
def criterion(y_pred, y_true, log_vars):
    loss = 0
    # method 1
    # for i in range(len(y_pred)):
    #     precision = torch.exp(-log_vars)
    #     diff = (y_pred - y_true) ** 2.0
    #     loss += torch.sum(precision * diff + log_vars, -1)
    # method 2
    # for i in range(len(y_pred)):
    #     precision = 0.5 / (log_vars ** 2)
    #     diff = (y_pred - y_true) ** 2.0
    #     loss += torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)
    precision = 0.5 / (log_vars ** 2)
    diff = (y_pred - y_true) ** 2.0
    # loss += torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)
    loss = torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)
    return torch.mean(loss)

def criterion_smooth(y_pred, y_true, log_vars):
    # print(y_pred.shape)
    loss = 0
    # method 1
    # for i in range(len(y_pred)):
    #     precision = torch.exp(-log_vars)
    #     diff = (y_pred - y_true) ** 2.0
    #     loss += torch.sum(precision * diff + log_vars, -1)
    # method 2
    # for i in range(len(y_pred)):
    #     precision = 0.5 / (log_vars ** 2)
    #     diff = (y_pred - y_true) ** 2.0
    #     loss += torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    precision = 0.5 / (log_vars ** 2)
    diff = 0
    Beta = 0.01
    for i in range(len(y_pred)):
        if np.abs(y_pred[i] - y_true[i]) < Beta:
            diff += 0.5 * (y_pred[i] - y_true[i]) ** 2.0 /Beta
        else:
            diff += np.abs(y_pred[i] - y_true[i])-0.5*Beta
    # loss += torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)
    diff = torch.from_numpy(diff)
    # 将所有张量移动到cuda:0设备上
    diff = diff.to('cuda:0')
    log_vars = log_vars.to('cuda:0')
    precision = precision.to('cuda:0')

    loss = torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)/len(y_pred)
    return torch.mean(loss)

# define cross loss criterion
def criterion_cross(y_pred, y_true, log_vars):

    precision = 0.5 / (log_vars ** 2)
    diff = (y_pred - y_true) ** 2.0
    # loss += torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)
    loss = torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)
    return torch.mean(loss)


iterations = 100000
min_loss = 10000000
time1 = time.time()
print(time1)
scaler = GradScaler()
for epoch in range(iterations):
    optimizer.zero_grad()  # to make the gradients zero

    # with autocast():

    # Loss based on boundary conditions
    predict_bc = net(torch.cat([x_bc, y_bc, t_bc],1))
    predict_u_bc, predict_v_bc, predict_p_bc, predict_a_bc = predict_bc[:, 0].reshape(-1, 1), predict_bc[:, 1].reshape(-1, 1), predict_bc[:, 2].reshape(-1, 1), predict_bc[:, 3].reshape(-1, 1)
    mse_u_bc = criterion(predict_u_bc, u_bc, log_var_m)
    mse_v_bc = criterion(predict_v_bc, v_bc, log_var_n)
    mse_p_bc = criterion(predict_p_bc, p_bc, log_var_o)
    mse_a_bc = criterion(predict_a_bc, a_bc, log_var_p)
    # Loss based on initial value
    predict_ic = net(torch.cat([x_ic, y_ic, t_ic], 1))
    predict_u_ic, predict_v_ic, predict_p_ic, predict_a_ic = predict_ic[:, 0].reshape(-1, 1), predict_ic[:, 1].reshape(-1, 1), predict_ic[:, 2].reshape(-1, 1), predict_ic[:, 3].reshape(-1, 1)
    mse_u_ic = criterion(predict_u_ic, u_ic, log_var_i)
    mse_v_ic = criterion(predict_v_ic, v_ic, log_var_j)
    mse_p_ic = criterion(predict_p_ic, p_ic, log_var_k)
    mse_a_ic = criterion(predict_a_ic, a_ic, log_var_l)

    # Loss based on PDE
    f_u, f_v, f_e, f_a = NS_bubble(x_pde, y_pde, t_pde, net=net)  # output of f(t,x)
    mse_f1 = criterion(f_u, torch.zeros_like(f_u), log_var_a)
    mse_f2 = criterion(f_v, torch.zeros_like(f_u), log_var_b)
    mse_f3 = criterion(f_e, torch.zeros_like(f_u), log_var_c)
    mse_f4 = criterion(f_a, torch.zeros_like(f_u), log_var_d)

    # Loss based on pde value
    predict_pde = net(torch.cat([x_pde, y_pde, t_pde], 1))
    predict_u_pde, predict_v_pde, predict_p_pde, predict_a_pde = predict_pde[:, 0].reshape(-1, 1), predict_pde[:,1].reshape(-1,1), predict_pde[:,2].reshape(-1, 1), predict_pde[:, 3].reshape(-1, 1)
    mse_u_pde = criterion(predict_u_pde, u_pde, log_var_e)
    mse_v_pde = criterion(predict_v_pde, v_pde, log_var_f)
    mse_p_pde = criterion(predict_p_pde, p_pde, log_var_g)
    mse_a_pde = criterion(predict_a_pde, a_pde, log_var_h)

    # Combining the loss functions
    loss = mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_u_pde + mse_v_pde + mse_p_pde + mse_a_pde + mse_u_bc + mse_v_bc + mse_p_bc + mse_a_bc + mse_u_ic + mse_v_ic + mse_p_ic + mse_a_ic



    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()

    loss.backward()  # This is for computing gradients using backward propagation
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    # scheduler.step(loss)

    scheduler.step()



    # torch.save(net, '/home/user/ZHOU-Wen/PINN/two-phase-flow/LS/0.6-1.2/last_net.pth')
    if loss < min_loss:
        min_loss = loss
        # 保存模型语句
        torch.save(net, '/home/user/ZHOU-Wen/PINN/two-phase-flow/LS/0.6-1.2/best_net_pde.pth')
        # torch.save(net, '/home/user/ZHOU-Wen/PINN/two-phase-flow/LS/0.6-1.2/best_net_resample_pde.pth')
        print('current epoch',epoch)
    with torch.autograd.no_grad():
        if epoch%1==0:
            data = [loss.item(),

                    # mse_f1.item(), mse_f2.item(), mse_f3.item(),mse_f4.item(),
                    # mse_u_bc.item(), mse_v_bc.item(), mse_p_bc.item(), mse_a_bc.item(),
                    # mse_u_ic.item(), mse_v_ic.item(), mse_p_ic.item(), mse_a_ic.item(),

                    mse_cost_function(f_u, torch.zeros_like(f_u)).item(),
                    mse_cost_function(f_v, torch.zeros_like(f_v)).item(),
                    mse_cost_function(f_e, torch.zeros_like(f_e)).item(),
                    mse_cost_function(f_a, torch.zeros_like(f_a)).item(),

                    mse_cost_function(predict_u_ic, u_ic).item(), mse_cost_function(predict_v_ic, v_ic).item(),
                    mse_cost_function(predict_p_ic, p_ic).item(), mse_cost_function(predict_a_ic, a_ic).item(),

                    mse_cost_function(predict_u_bc, u_bc).item(), mse_cost_function(predict_v_bc, v_bc).item(),
                    mse_cost_function(predict_p_bc, p_bc).item(), mse_cost_function(predict_a_bc, a_bc).item(),

                    mse_cost_function(predict_u_pde, u_pde).item(), mse_cost_function(predict_v_pde, v_pde).item(),
                    mse_cost_function(predict_p_pde, p_pde).item(), mse_cost_function(predict_a_pde, a_pde).item(),

                    log_var_a.item(), log_var_b.item(), log_var_c.item(), log_var_d.item(),
                    log_var_e.item(), log_var_f.item(), log_var_g.item(), log_var_h.item(),
                    log_var_i.item(), log_var_j.item(), log_var_k.item(), log_var_l.item(),
                    log_var_m.item(), log_var_n.item(), log_var_o.item(), log_var_p.item(), ]

            with open(loss_path, 'a', encoding='utf-8', newline='') as fp:
                # 写
                writer = csv.writer(fp)
                # 将数据写入
                writer.writerow(list(data))
        if epoch%100==0:
            # print(epoch, "Traning Loss:", loss.data)
            print(epoch, "Traning Loss:", loss.item())

time2 = time.time()
print(time2)
print('userd: {:.5f}s'.format(time2-time1))

