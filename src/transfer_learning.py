import torch
import numpy as np
from torch import nn
from net import Net
from matplotlib import pyplot as plt
from train import setup_seed
from train import get_train_test
from metric import evaluation


window_size = 300
epochs = 1500
lr = 0.001
hidden_dim = 256
num_layers = 2
weight_decay = 0.0
rated_capacity = 1.1

device = torch.device('cpu')

name_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
datasource = 'D:/pycharm_workspace/my_dachuang/dataset/data_2.npy'
datas = np.load(datasource, allow_pickle=True).item()


def show_parm(m_model):
    for name, param in m_model.named_parameters():
        print(name, param.shape)
        print(param)
        print()


def freeze_parm(m_model):
    """
    冻结除全连接层以外层的参数
    :param m_model: 模型
    """
    for name, param in m_model.named_parameters():
        if 'linear' not in name:
            param.requires_grad = False


def transfer_learning(b_name, se):
    """
    将某一个模型迁移到其他数据集上，例如将CS2_36迁移到CS2_35,CS2_37
    :param b_name: 待迁移模型的名称
    :param se: 种子
    """
    print('====================' + b_name + '====================')

    full_path = '../model/lstm_full_parm_' + b_name + '.pth'
    model = Net(input_size=window_size, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    state_dict = torch.load(full_path)
    model.load_state_dict(state_dict)
    freeze_parm(model)

    for name in name_list:
        if name == 'CS2_35' or name == 'CS2_37' or name == 'CS2_38':
            continue

        print(b_name + ' is transferred to predict ' + name)
        train_x_sequence, train_y_sequence, train_data, test_data = get_train_test(name)
        train_size = len(train_x_sequence)
        print('sample size: {}'.format(train_size))

        setup_seed(se)
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

            if (epoch + 1) % 10 == 0:
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
                if mae < 0.02:
                    break
                print(
                    'epoch:{:<2d} | loss:{:<6.4f} | MAE:{:<6.4f} |  MSE:{:<6.4f} | RMSE:{:<6.4f}'.format(
                        epoch, loss, mae, mse, rmse))
                score = [mae, mse, rmse]
                score_lists.append(score)

        torch.save(model, '../model/lstm_' + b_name + 'tl_' + name + '.pth')
        torch.save(model.state_dict(), '../model/lstm_parm_' + b_name + 'tl_' + name + '.pth')
        print()


def test(path, b_name):
    for name in name_list:
        if name == 'CS2_35' or name == 'CS2_36' or name == 'CS2-37':
            continue

        print('load model ' + path)
        model = torch.load(path)

        data = np.array(datas[b_name]['Capacity'])
        example_data = list(data[:window_size])
        test_x = list(example_data)
        target_data = list(data[window_size:])
        point_list = []

        while (len(test_x) - len(example_data)) < len(target_data):
            x = np.reshape(np.array(test_x[-window_size:]) / rated_capacity, (-1, 1, window_size)).astype(
                np.float32)
            x = torch.from_numpy(x).to(device)  # shape: (batch_size, 1, input_size)
            pred = model(x)
            next_point = pred.data.numpy()[0, 0] * rated_capacity
            test_x.append(next_point)  # 测试值加入原来序列用来继续预测下一个点
            point_list.append(next_point)  # 保存输出序列最后一个点的预测值

        mae, mse, rmse = evaluation(y_test=target_data, y_predict=point_list)
        print('MAE:{:<6.4f} |  MSE:{:<6.4f} | RMSE:{:<6.4f}'.format(mae, mse, rmse))
        # visualize(b_name, name, data, np.array(point_list))

        return data, np.array(point_list)


def visualize(b_name, name, data, predict_data):
    plt.figure(figsize=(12, 9))
    plt.xlabel('Cycle')
    plt.ylabel('Capacity(Ah)')
    plt.scatter(np.linspace(1, len(data), len(data)), data, s=10, marker='.', c='b', linewidths=1)
    plt.scatter(np.linspace(len(data) - len(predict_data), len(data), len(predict_data)), predict_data, s=10,
                marker='.', c='r', linewidths=1)
    plt.savefig('../dataset/visualize/eol_predict/tl_model_' + b_name + 'predict' + name + '.png')
    plt.close()


def final_visualize(b_name, name, t_data, t_predict_data, f_data, f_predict_data):
    plt.figure(figsize=(12, 9))
    plt.xlabel('Cycle')
    plt.ylabel('Capacity(Ah)')
    plt.scatter(np.linspace(1, len(t_data), len(t_data)), t_data, s=10, marker='.', c='b', linewidths=1)
    plt.scatter(np.linspace(len(t_data) - len(t_predict_data), len(t_data), len(t_predict_data)), t_predict_data, s=10,
                marker='.', c='r', linewidths=1)
    plt.scatter(np.linspace(len(f_data) - len(f_predict_data), len(f_data), len(f_predict_data)), f_predict_data, s=10,
                marker='.', c='g', linewidths=1)
    plt.savefig('../dataset/visualize/eol_predict/tl&full_model_' + b_name + 'predict' + name + '.png')
    plt.close()


if __name__ == '__main__':
    seed = 6
    for battery_name in name_list:
        if battery_name == 'CS2_35':
            continue

        tl_path = '../model/lstm_' + 'CS2_38' + 'tl_' + battery_name + '.pth'
        full_path = '../model/lstm_full_' + battery_name + '.pth'
        tl_data, tl_predict_data = test(tl_path, battery_name)
        full_data, full_predict_data = test(full_path, battery_name)
        final_visualize('CS2_38', battery_name, tl_data, tl_predict_data, full_data, full_predict_data)

        # print('********************************* transfer learning starts *********************************')
        # transfer_learning(battery_name, seed)
        #
        # print('********************************* test starts *********************************')
        # test(battery_name)
