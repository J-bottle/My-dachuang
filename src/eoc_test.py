import torch
import numpy as np
from matplotlib import pyplot as plt
from metric import evaluation

from eoc_train import train

model_path = '../model/lstm1.pt'
name_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
datasource = 'D:/pycharm_workspace/my_dachuang/dataset/data_2.npy'
datas = np.load(datasource, allow_pickle=True).item()
device = torch.device('cpu')

window_size = 50
hidden_dim = 128
num_layers = 2
weight_decay = 0.0

rated_voltage = 4.0


def visualize(m_name, cycle, voltage_list, time_list, voltage_predict_list):
    pivot = len(time_list) - len(voltage_predict_list)
    plt.figure(figsize=(12, 9))
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage(V)')
    plt.scatter(time_list, voltage_list, s=10, marker='.', c='b', linewidths=1)
    plt.scatter(time_list[pivot:], voltage_predict_list, s=10, marker='.', c='r', linewidths=1)
    plt.savefig(
        '../dataset/visualize/eoc_predict_test/model_' + m_name + 'predict_' + str(cycle) + '.png')
    plt.close()


def test(m_name, voltage_list, time_list):
    model = torch.load('../model/lstm_eoc_' + m_name + '.pth')

    print('******************************** test ' + m_name + ' begins ********************************')

    example_data = list(voltage_list[:window_size])
    test_x = list(example_data)
    target_data = list(voltage_list[window_size:])
    point_list = []

    while (len(test_x) - len(example_data)) < len(target_data):
        x = np.reshape(np.array(test_x[-window_size:]) / rated_voltage, (-1, 1, window_size)).astype(
            np.float32)
        x = torch.from_numpy(x).to(device)  # shape: (batch_size, 1, input_size)
        pred = model(x)
        next_point = pred.data.numpy()[0, 0] * rated_voltage
        test_x.append(next_point)  # 测试值加入原来序列用来继续预测下一个点
        point_list.append(next_point)  # 保存输出序列最后一个点的预测值

    mae, mse, rmse = evaluation(y_test=target_data, y_predict=point_list)
    print('MAE:{:<6.4f} |  MSE:{:<6.4f} | RMSE:{:<6.4f}'.format(mae, mse, rmse))
    print('visualize ' + str(1) + ' cycle voltage')
    visualize(m_name, 1, voltage_list, time_list, point_list)


if __name__ == '__main__':
    seed = 6
    v_list, t_list = train(seed)
    test('CS2_36', v_list, t_list)
