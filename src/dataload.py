import pandas as pd
import glob
import numpy as np

datasource = 'D:/pycharm_workspace/my_dachuang/dataset/'
battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']


def read_data():
    battery_data = {}
    for name in battery_list:
        print('*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=')
        print(name)
        directory = datasource + name + '/*.xlsx'
        path = glob.glob(directory)
        cnt = 0  # 记录总cycle数
        # 注意这种写法是共用内存的！v_list = v_diff_list = date_list = cap_list = ir_list = ccct_list = ccvt_list = []
        v_list = []
        v_diff_list = []
        date_list = []
        cap_list = []
        ir_list = []
        ccct_list = []
        ccvt_list = []
        for file in path:
            print('Load ' + file + '...')
            df = pd.read_excel(file, sheet_name=1)
            cycles = list(set(df['Cycle_Index']))
            cnt += len(cycles)
            for c in cycles:
                df_c = df[df['Cycle_Index'] == c]  # 所有循环号为C的数据
                '''End of charge data'''
                df_c_eoc = df_c[df_c['Step_Index'] == 7]
                if df_c_eoc.empty:
                    cnt -= 1
                    continue
                df_c_v = df_c_eoc['Voltage(V)']
                df_c_v_diff = df_c_eoc['dV/dt(V/s)']
                df_c_date = df_c_eoc['Test_Time(s)']

                df_c_date = date_norm(df_c_date)
                v_list.append(list(df_c_v))
                v_diff_list.append(list(df_c_v_diff))
                date_list.append(df_c_date)

                '''End of life data'''
                df_c_eol = df_c[df_c['Step_Index'] == 7]
                capacity = list(df_c_eol['Discharge_Capacity(Ah)'])[-1]
                ir = list(df_c_eol['Internal_Resistance(Ohm)'])[0]
                cap_list.append(capacity)
                ir_list.append(ir)

                df_c_ccct = df_c[df_c['Step_Index'] == 2]
                df_c_ccvt = df_c[df_c['Step_Index'] == 4]
                ccct_time_list = list(df_c_ccct['Test_Time(s)'])
                ccvt_time_list = list(df_c_ccvt['Test_Time(s)'])

                if ccct_time_list:
                    ccct_time_list = date_norm(ccct_time_list)
                    ccct_list.append(max(ccct_time_list))
                else:
                    ccct_list.append(0.0)
                if ccvt_time_list:
                    ccvt_time_list = date_norm(ccvt_time_list)
                    ccvt_list.append(max(ccvt_time_list))
                else:
                    ccvt_list.append(0.0)

        index = np.linspace(1, cnt, cnt)
        # df_result = pd.DataFrame(
        #     {'Cycles': index, 'Date': date_list[index], 'Voltage': v_list[index],
        #      'Voltage_differential': v_diff_list[index], 'Capacity': cap_list[index], 'Resistance': ir_list[index],
        #      'CCCT': ccct_list[index], 'CCVT': ccvt_list[index]})
        df_result = pd.DataFrame(
            {'Cycles': index, 'Date': date_list, 'Voltage': v_list, 'Voltage_differential': v_diff_list,
             'Capacity': cap_list, 'Resistance': ir_list, 'CCCT': ccct_list, 'CCVT': ccvt_list})
        battery_data[name] = df_result

    return battery_data


def date_norm(df_time):
    times = np.array(df_time)
    if len(times) > 1:
        times -= times[0]
    return list(times)


if __name__ == '__main__':
    data = read_data()
    # np.save('../dataset/data.npy', data)
    np.save('../dataset/data_1.npy', data)  # 每组时间调整为从0.0开始
