import torch
import torch.nn as nn
import numpy as np
import random
import os

from matplotlib import pyplot as plt
from net import Net
from metric import evaluation

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
name_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
datasource = 'D:/pycharm_workspace/my_dachuang/dataset/data_2.npy'
datas = np.load(datasource, allow_pickle=True).item()

window_size = 50
epochs = 10000
lr = 0.0005
hidden_dim = 256
num_layers = 2
weight_decay = 0.0
rated_voltage = 4.0
dt = 10  # 左右各差值，即三等分点都插值


def setup_seed(se):
    np.random.seed(se)  # Numpy module.
    random.seed(se)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(se)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(se)  # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(se)  # 为当前GPU设置随机种子
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def interpolation(name, d_t):
    """
    利用线性插值的方式补充，使得应受重视的点的数量更多
    :param d_t: 插值的微分间隔
    :param name: 型号，例如：CS2_35,CS2_36,...
    :return: 返回经过插值后的电压数据
    """
    voltage_diff_list = datas[name]['Voltage_differential']
    voltage_list = datas[name]['Voltage']
    times = datas[name]['Date']

    voltage_diff = np.array(voltage_diff_list[0])
    voltage = np.array(voltage_list[0])
    time = times[0]

    index = np.where(voltage < 3.4)

    voltage = list(voltage)
    voltage_diff = list(voltage_diff)

    pivot = index[0][0]
    voltage_still = voltage[:pivot].copy()
    time_still = time[:pivot].copy()

    '''注意d_v为负数！'''
    for i in range(len(index[0])):
        idx = index[0][i]
        v = voltage[idx]
        t = time[idx]
        d_v = voltage_diff[idx] * d_t
        if i == 0:
            voltage_still.append(v)
            voltage_still.append(v + d_v / 2)
            voltage_still.append(v + d_v)
            time_still.append(t + d_t / 2)
            time_still.append(t + d_t)
            time_still.append(t)

        elif i == len(index[0]) - 1:
            voltage_still.append(v - d_v)
            voltage_still.append(v - d_v/2)
            voltage_still.append(v)
            time_still.append(t)
            time_still.append(t - d_t/2)
            time_still.append(t - d_t)
        else:
            voltage_still.append(v - d_v)
            voltage_still.append(v - d_v / 2)
            voltage_still.append(v)
            voltage_still.append(v + d_v / 2)
            voltage_still.append(v + d_v)
            time_still.append(t - d_t)
            time_still.append(t - d_t/2)
            time_still.append(t)
            time_still.append(t + d_t/2)
            time_still.append(t + d_t)

    voltage_interpolation = voltage_still.copy()
    time_interpolation = time_still.copy()

    visualize(voltage_interpolation, time_interpolation)

    return voltage_interpolation, time_interpolation


def get_train_test(voltage_list):
    train_sequence = []
    target_sequence = []

    voltage = np.array(voltage_list)

    train_data = voltage[:window_size]
    test_data = voltage[window_size:]

    for index in range(len(voltage) - window_size):
        train_sequence.append(voltage[index:index + window_size])
        target_sequence.append(voltage[index + window_size])

    return np.array(train_sequence), np.array(target_sequence), list(train_data), list(test_data)


def train(se):
    """
    仅仅使用'CS2_37'数据集训练，将训练好的模型迁移到CS2_35,CS2_37,CS2_38上
    :param se: 随机种子
    """
    name = 'CS2_36'
    voltage_list, time_list = interpolation(name, dt)
    train_x_sequence, train_y_sequence, train_data, test_data = get_train_test(voltage_list)
    train_size = len(train_x_sequence)
    print('sample size: {}'.format(train_size))

    setup_seed(se)
    model = Net(input_size=window_size, hidden_dim=hidden_dim, num_layers=num_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        X = np.reshape(train_x_sequence / rated_voltage, (-1, 1, window_size)).astype(
            np.float32)  # (batch_size, seq_len, input_size)
        y = np.reshape(train_y_sequence / rated_voltage, (-1, 1)).astype(np.float32)  # shape 为 (batch_size, 1)

        X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
        output = model(X)
        output = output.reshape(-1, 1)
        loss = criterion(output, y)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        loss_list, y_ = [0], []
        # if epoch == 1499:
        if (epoch + 1) % 10 == 0:
            test_x = train_data.copy()  # 每100次重新预测一次
            point_list = []
            while (len(test_x) - len(train_data)) < len(test_data):
                x = np.reshape(np.array(test_x[-window_size:]) / rated_voltage, (-1, 1, window_size)).astype(
                    np.float32)
                x = torch.from_numpy(x).to(device)  # shape: (batch_size, 1, input_size)
                pred = model(x)
                next_point = pred.data.numpy()[0, 0] * rated_voltage
                test_x.append(next_point)  # 测试值加入原来序列用来继续预测下一个点
                point_list.append(next_point)  # 保存输出序列最后一个点的预测值
            y_.append(point_list)  # 保存本次预测所有的预测值
            loss_list.append(loss)

            mae, mse, rmse = evaluation(y_test=test_data, y_predict=y_[-1])
            print(
                'epoch:{:<2d} | loss:{:<6.4f} | MAE:{:<6.4f} |  MSE:{:<6.4f} | RMSE:{:<6.4f}'.format(
                    epoch, loss, mae, mse, rmse))

            # debug_visualize(voltage_list, time_list, point_list)
            if mse < 0.001:
                break
    torch.save(model, '../model/lstm_eoc_' + name + '.pth')
    torch.save(model.state_dict(), '../model/lstm_eoc_parm_' + name + '.pth')

    return voltage_list, time_list


def visualize(voltage_list, time_list):
    plt.figure(figsize=(12, 9))
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage(V)')
    plt.scatter(time_list, voltage_list, s=10, marker='.', c='b', linewidths=1)
    plt.show()
    plt.close()


def debug_visualize(voltage_list, time_list, voltage_predict_list):
    pivot = len(time_list) - len(voltage_predict_list)
    plt.figure(figsize=(12, 9))
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage(V)')
    plt.scatter(time_list, voltage_list, s=10, marker='.', c='b', linewidths=1)
    plt.scatter(time_list[pivot:], voltage_predict_list, s=10, marker='.', c='r', linewidths=1)
    plt.show()
    plt.close()


if __name__ == '__main__':
    seed = 6
    train(6)
