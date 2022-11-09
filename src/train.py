import torch
import torch.nn as nn
import numpy as np
import random
import os

from net import Net
from metric import evaluation, relative_error

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
name_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
datasource = 'D:/pycharm_workspace/my_dachuang/dataset/data_2.npy'
datas = np.load(datasource, allow_pickle=True).item()

window_size = 300
epochs = 1000
lr = 0.001
hidden_dim = 256
num_layers = 2
weight_decay = 0.0
rated_capacity = 1.1


def setup_seed(se):
    np.random.seed(se)  # Numpy module.
    random.seed(se)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(se)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(se)  # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(se)  # 为当前GPU设置随机种子
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_train_test(name):
    battery = datas[name]
    capacity = battery['Capacity']
    capacity = np.array(capacity)
    train_data = capacity[:window_size]
    test_data = capacity[window_size:]

    train_sequence = []
    target_sequence = []
    for index in range(len(capacity) - window_size):
        train_sequence.append(capacity[index:index + window_size])
        target_sequence.append(capacity[index + window_size])

    return np.array(train_sequence), np.array(target_sequence), list(train_data), list(test_data)


def train(se):
    """
    仅仅使用'CS2_36'数据集训练，将训练好的模型迁移到CS2_35,CS2_37,CS2_38上
    :param se: 随机种子
    :return:
    """
    name = 'CS2_38'
    train_x_sequence, train_y_sequence, train_data, test_data = get_train_test(name)
    train_size = len(train_x_sequence)
    print('sample size: {}'.format(train_size))

    setup_seed(se)
    model = Net(input_size=window_size, hidden_dim=hidden_dim, num_layers=num_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    loss_list, y_ = [0], []
    score_lists = []
    for epoch in range(epochs):
        X = np.reshape(train_x_sequence / rated_capacity, (-1, 1, window_size)).astype(
            np.float32)  # (batch_size, seq_len, input_size)
        y = np.reshape(train_y_sequence / rated_capacity, (-1, 1)).astype(np.float32)  # shape 为 (batch_size, 1)

        X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
        output = model(X)
        output = output.reshape(-1, 1)
        loss = criterion(output, y)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if (epoch + 1) % 100 == 0:
            test_x = train_data.copy()  # 每100次重新预测一次
            point_list = []
            while (len(test_x) - len(train_data)) < len(test_data):
                x = np.reshape(np.array(test_x[-window_size:]) / rated_capacity, (-1, 1, window_size)).astype(
                    np.float32)
                x = torch.from_numpy(x).to(device)  # shape: (batch_size, 1, input_size)
                pred = model(x)
                next_point = pred.data.numpy()[0, 0] * rated_capacity
                test_x.append(next_point)  # 测试值加入原来序列用来继续预测下一个点
                point_list.append(next_point)  # 保存输出序列最后一个点的预测值
            y_.append(point_list)  # 保存本次预测所有的预测值
            loss_list.append(loss)
            mae, mse, rmse = evaluation(y_test=test_data, y_predict=y_[-1])
            re = relative_error(y_test=test_data, y_predict=y_[-1], threshold=rated_capacity * 0.7)
            print(
                'epoch:{:<2d} | loss:{:<6.4f} | MAE:{:<6.4f} |  MSE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(
                    epoch, loss, mae, mse, rmse, re))
            score = [mae, mse, rmse]
            score_lists.append(score)

    torch.save(model, '../model/lstm_full_' + name + '.pth')
    torch.save(model.state_dict(), '../model/lstm_full_parm_' + name + '.pth')

    return score_lists


if __name__ == '__main__':
    seed = 6
    score_list = train(6)
    print(score_list)
