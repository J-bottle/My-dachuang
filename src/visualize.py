import os.path

import numpy as np
import matplotlib.pyplot as plt
from datasift import array_diff

datasource1 = 'D:/pycharm_workspace/my_dachuang/dataset/data_1.npy'
data1 = np.load(datasource1, allow_pickle=True).item()
datasource2 = 'D:/pycharm_workspace/my_dachuang/dataset/data_2.npy'
data2 = np.load(datasource2, allow_pickle=True).item()
name_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
batch_index = {'CS2_35': 236, 'CS2_36': 726, 'CS2_37': 790, 'CS2_38': 782}
new_batch_index = {'CS2_35': 225, 'CS2_36': 712, 'CS2_37': 771, 'CS2_38': 755}  # 经过datasift后的有用下标


def visualize_eol(name, cap, resist, ccct_, ccvt_):
    feature = [cap, resist, ccct_, ccvt_]
    plt.figure(figsize=(12, 9))
    y_names = ['Capacity(Ah)', 'Resistance(Ohm)', 'CCCT(s)', 'CCVT(s)']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.scatter(np.linspace(1, len(cap), len(cap)), feature[i], s=10, marker='.', linewidths=1)
        # plt.plot(np.linspace(1, len(cap), len(cap)), feature[i])
        plt.xlabel('Cycles', fontsize=14)
        plt.ylabel(y_names[i], fontsize=14)

    plt.savefig('../dataset/visualize/final/' + name + '_eol.png')
    plt.close()


def visualize_eol_sift(name, cap, cap_sift, resist, resist_sift, ccct_, ccct_sift, ccvt_, ccvt_sift):
    """
    所有带_sift的都是为了筛选而先做可视化的，决定阈值用于排除孤立点
    """
    feature = [array_diff(cap), resist, ccct_, ccvt_]
    feature_sift = [array_diff(cap_sift) - 0.08, resist_sift, ccct_sift, ccvt_sift]
    plt.figure(figsize=(12, 9))
    y_names = ['Capacity(Ah)', 'Resistance(Ohm)', 'CCCT(s)', 'CCVT(s)']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.scatter(np.linspace(1, len(cap), len(cap)), feature[i], s=10, marker='.', c='b')
        plt.scatter(np.linspace(1, len(cap), len(cap)), feature_sift[i], s=10, marker='.', c='r')
        plt.xlabel('Cycles', fontsize=14)
        plt.ylabel(y_names[i], fontsize=14)

    plt.savefig('../dataset/visualize/sift/' + name + '_eol.png')
    plt.close()


def visualize_eoc(name, vol, vol_diff, dates):
    for i in range(len(vol)):
        path = '../dataset/visualize/eoc/' + name + '_cycle_' + str(i + 1) + '_eoc.png'
        if os.path.exists(path):
            continue
        vol_list = vol[i]
        vol_diff_list = vol_diff[i]
        date_list = dates[i]
        feature = [vol_list, vol_diff_list]
        y_names = ['Voltage(V)', 'Voltage_differential']
        plt.figure(figsize=(12, 9))
        for j in range(len(feature)):
            plt.subplot(1, 2, j + 1)
            plt.scatter(date_list, feature[j], s=10, marker='.', linewidths=1)
            plt.xlabel('Time(s)', fontsize=14)
            plt.ylabel(y_names[j], fontsize=14)

        print('Save picture ' + str(i + 1))
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    for it in name_list:
        print(it)
        name_data = data2[it]
        cycle = np.array(name_data['Cycles'])
        date = np.array(name_data['Date'])
        voltage = np.array(name_data['Voltage'])
        voltage_differential = np.array(name_data['Voltage_differential'])
        capacity = np.array(name_data['Capacity'])
        resistance = np.array(name_data['Resistance'])
        ccct = np.array(name_data['CCCT'])
        ccvt = np.array(name_data['CCVT'])

        # visualize_eol(it + '_1', capacity[:batch_index[it]], resistance[:batch_index[it]], ccct[:batch_index[it]],
        #               ccvt[:batch_index[it]])
        # visualize_eol(it + '_2', capacity[batch_index[it]:], resistance[batch_index[it]:], ccct[batch_index[it]:],
        #               ccvt[batch_index[it]:])

        # visualize_eol_sift(it + '_1', capacity[:batch_index[it]], capacity[:batch_index[it]].copy(),
        #                    resistance[:batch_index[it]], resistance[:batch_index[it]] + 0.01,
        #                    ccct[:batch_index[it]], ccct[:batch_index[it]] - 500,
        #                    ccvt[:batch_index[it]], ccvt[:batch_index[it]] + 5000)

        # visualize_eol(it, capacity[:new_batch_index[it]], resistance[:new_batch_index[it]],ccct[:new_batch_index[it]],
        #               ccvt[:new_batch_index[it]])

        visualize_eoc(it, voltage[:new_batch_index[it]], voltage_differential[:new_batch_index[it]],
                      date[:new_batch_index[it]])
