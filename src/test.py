import torch
import numpy as np
from matplotlib import pyplot as plt
from metric import evaluation
from metric import relative_error

model_path = '../model/lstm1.pt'
name_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
datasource = 'D:/pycharm_workspace/my_dachuang/dataset/data_2.npy'
datas = np.load(datasource, allow_pickle=True).item()
device = torch.device('cpu')

window_size = 300
hidden_dim = 128
num_layers = 2
weight_decay = 0.0

rated_capacity = 1.1


def visualize(m_name, battery, data, predict_data):
    plt.figure(figsize=(12, 9))
    plt.xlabel('Cycle')
    plt.ylabel('Capacity(Ah)')
    plt.scatter(np.linspace(1, len(data), len(data)), data, s=10, marker='.', c='b', linewidths=1)
    plt.scatter(np.linspace(len(data) - len(predict_data), len(data), len(predict_data)), predict_data, s=10,
                marker='.', c='r', linewidths=1)
    plt.savefig('../dataset/visualize/eol_predict/model_' + m_name + 'predict_' + battery + '.png')


def test(m_name):
    model = torch.load('../model/lstm_full_' + m_name + '.pth')
    for battery_name in name_list:
        if battery_name == 'CS2_35':
            continue
        data = np.array(datas[battery_name]['Capacity'])
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
        re = relative_error(y_test=target_data, y_predict=point_list, threshold=rated_capacity * 0.7)
        print('MAE:{:<6.4f} |  MSE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(mae, mse, rmse, re))
        visualize(m_name, battery_name, data, np.array(point_list))


if __name__ == '__main__':
    for model_name in name_list:
        if model_name == 'CS2_35':
            continue
        test(model_name)
        print()
