import numpy as np

datasource = 'D:/pycharm_workspace/my_dachuang/dataset/data_1.npy'
data = np.load(datasource, allow_pickle=True).item()
idx = 286


def drop_outlier(array, count, bins):
    """
    :param array: 需要处理的数组
    :param count: 数组的长度（一维）
    :param bins: 每次drop的数组长度
    :return: 所有合理的元素下标列表
    """
    index = []
    range_ = np.arange(0, count, bins)
    for i in range_[:]:  # [:-1]不包括最后一个元素
        array_lim = array[i:i + bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max, th_min = mean + sigma * 3, mean - sigma * 3
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)


def reverse_search():
    array = np.array([1, 2, 3, 4, 5])
    for i in range(len(array) - 1, 0, -1):
        print(i)


if __name__ == '__main__':
    reverse_search()
    a = np.array(data['CS2_35']['Resistance'])
    index1 = drop_outlier(a[:idx], idx, 40)
    index2 = drop_outlier(a[idx:], len(a) - idx, 40) + idx
    index = np.append(index1, index2)
    print()
