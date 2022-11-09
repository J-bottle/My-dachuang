import numpy as np
import pandas as pd
from functools import reduce

datasource = 'D:/pycharm_workspace/my_dachuang/dataset/data_1.npy'
data = np.load(datasource, allow_pickle=True).item()
name_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
batch_index = {'CS2_35': 236, 'CS2_36': 726, 'CS2_37': 790, 'CS2_38': 782}


def data_sift():
    battery_data = {}
    for it in name_list:

        print(it)
        name_data = data[it]

        '''EOL sift'''
        date = np.array(name_data['Date'])
        voltage = np.array(name_data['Voltage'])
        vol_diff = np.array(name_data['Voltage_differential'])
        capacity = np.array(name_data['Capacity'])
        resistance = np.array(name_data['Resistance'])
        ccct = np.array(name_data['CCCT'])
        ccvt = np.array(name_data['CCVT'])
        capacity = array_diff(capacity)
        cap_index, len1 = sift_by_type(it, capacity.copy(), 'capacity')
        if it == 'CS2_35':
            del_index = np.where((cap_index == 210) | (cap_index == 211))  # 还有两个明显的点，干脆手动删除...
            cap_index = np.delete(cap_index, del_index[0])
            len1 -= 2
        elif it == 'CS2_36':
            del_index = np.where(cap_index == 429)  # 还有一个明显的点，干脆手动删除...
            cap_index = np.delete(cap_index, del_index[0])
            len1 -= 1

        resist_index, len2 = sift_by_type(it, resistance, 'resist')
        ccct_index, len3 = sift_by_type(it, ccct, 'ccct')
        ccvt_index, len4 = sift_by_type(it, ccvt, 'ccvt')
        index = reduce(np.intersect1d, [cap_index, resist_index, ccct_index, ccvt_index])
        index_set = set(index)
        index_pd_del = list(((set(np.arange(len(date)))) ^ index_set))  # 通过取异或值得到dataframe中需要删除的行

        batch_index[it] = min(len1, len2, len3, len4)

        '''EOC sift'''
        # cnt = 0
        # for v_l in voltage:
        #     for i in range(len(v_l) - 1):
        #         if v_l[i] < v_l[i + 1]:
        #             cnt += 1
        # print(cnt)  # 0,0,0,2 故绝大部分上都是递减的电压数据，故EOC的数据不做筛选

        df_result = pd.DataFrame(
            {'Cycles': np.linspace(1, len(name_data), len(name_data)), 'Date': list(date), 'Voltage': list(voltage),
             'Voltage_differential': list(vol_diff),
             'Capacity': list(capacity), 'Resistance': list(resistance), 'CCCT': list(ccct), 'CCVT': list(ccvt)})

        '''删除dataframe中不符合要求的行'''
        df_result = df_result.drop(df_result.index[index_pd_del])
        battery_data[it] = df_result

    print(batch_index)
    return battery_data


def array_diff(array):
    """
    :param array: capacity
    :return: 因为原excel中capacity是累增的，所以返回取差分后的数组
    """
    for i in range(len(array) - 1, 0, -1):
        if array[i] > array[i - 1]:
            array[i] = array[i] - array[i - 1]

    return array


def sift_by_type(name, array, t):
    """
    一次的大小由bins决定
    :param name: 数据集名字：CS2_35,CS2_36,...
    :param array: 需要处理的不同物理量：capacity, resistance, ccct, ccvt
    :param t: 'capacity', 'resistance' 等
    :return: 返回有用数据的下标：index 和新的batch_index对应元素的值
    """
    i = batch_index[name]
    index = drop_outlier(array[:i], t)

    return index, len(index)


def drop_outlier(array, t):
    """
    显然像Capacity和Resistance这种本应递减或者递增的数据不可能服从正态分布，
    所以像有些博主写的用2sigma 或者 3sigma规则过滤异常点简直就是扯淡
    所以我认为先做个数据可视化，直观上感受下范围
    最后发现不同数据选用不同的筛选方法
    :param array: 需要处理的数组
    :param t: 'capacity', 'resistance' 等
    :return: 所有合理的元素下标列表
    """
    if t == 'ccct':
        index = drop_by_neighbors(array, 500)
    elif t == 'ccvt':
        index = drop_by_neighbors(array, 5000)

    elif t == 'capacity':
        index = drop_by_neighbors(array, 0.08)

    else:  # 内阻不太好过滤
        index = np.arange(len(array))
        pass

    print(t + '共过滤点' + str(len(array) - len(index)) + '个')
    return index


def drop_by_neighbors(array, theta):
    index = []
    for i in range(len(array)):
        if i == 0 and abs(array[i] - array[i + 1]) > theta:
            continue
        elif i == len(array) - 1 and abs(array[i] - array[i - 1]) > theta:
            continue
        elif abs(array[i] - array[i - 1]) > theta and abs(array[i] - array[i + 1]) > theta:
            continue
        else:
            index.append(i)

    index = np.array(index)

    return index


if __name__ == '__main__':
    data_sifted = data_sift()
    np.save('../dataset/data_2.npy', data_sifted)  # 删除了部分不合理的点
