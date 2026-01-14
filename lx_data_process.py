import pickle
import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import PchipInterpolator
from torch.utils.data import DataLoader, TensorDataset
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载OpenMP库
def dereference_h5_refs(file_path, dataset_name):
    """解引用HDF5对象引用数组，返回具体数据"""
    with h5py.File(file_path, "r") as f:
        ref_array = f[dataset_name][:]
        dereferenced = []

        for ref in ref_array.flatten():
            try:
                obj = f[ref]
                if isinstance(obj, h5py.Group):
                    # 解引用组中的数据集
                    group_data = {}
                    for key in obj.keys():
                        if isinstance(obj[key], h5py.Dataset):
                            group_data[key] = obj[key][:]
                    dereferenced.append({"type": "group", "name": obj.name, "data": group_data})
                elif isinstance(obj, h5py.Dataset):
                    data = obj[:]
                    if obj.dtype.kind == 'u' and obj.dtype.itemsize == 2:
                        letters = [chr(value[0]) for value in data]
                        data = ''.join(letters)
                    dereferenced.append({"type": "dataset", "name": obj.name, "data": data})
                else:
                    dereferenced.append({"type": "unknown"})
            except Exception as e:
                dereferenced.append({"error": str(e)})

        return dereferenced

def dereference_h5_spike(file_path, dataset_name):
    """解引用 HDF5 对象引用数组"""
    with h5py.File(file_path, "r") as f:
        ref_array = f[dataset_name][:].T
        dereferenced = []

        for ref in ref_array:
            trial=[]
            for channel in ref:
                try:
                    obj = f[channel]
                    if isinstance(obj, h5py.Dataset):
                        data = obj[:]
                        trial.append(data)
                    elif isinstance(obj, h5py.Group):
                        trial.append({"type": "group", "name": obj.name})
                    else:
                        trial.append({"type": "unknown"})
                except Exception as e:
                    trial.append({"error": str(e)})
            dereferenced.append(trial)

        return dereferenced

def reconstruct_spike_matrix(spike_time,trial_start_time,trial_end_time,time_interval,channels=64):

    spike_matrix = np.zeros((channels, time_interval + 1), dtype=int)

    for channel in range(channels):
        # 检查是否有脉冲发放
        if spike_time[channel].all() == 0:
            spike_matrix[channel][:] = 0
        elif spike_time[channel].all() != 0:
            # 有脉冲发放，按索引填充脉冲矩阵
            spike_time_point = spike_time[channel]
            for i in range(len(spike_time_point)):
                if trial_start_time <= spike_time_point[i] <= trial_end_time:
                    # 直接填充到对应位置
                    time_index = int(spike_time_point[i] - trial_start_time)
                    spike_matrix[channel][time_index] += 1
    print("(Sample rate 1 us) spike matrix shape ：",spike_matrix.shape)
    return spike_matrix


def pchip_interpolation(x, y):
    """
    对输入数据进行去重、排序，并使用pchip插值生成指定数量的插值点。

    参数:
    x (array-like): 输入的x值数组。
    y (array-like): 输入的y值数组，与x一一对应。

    返回:
    xInterp (numpy.ndarray): 插值后的x值数组。
    pchipInterp (numpy.ndarray): 插值后的y值数组。
    x_sorted (numpy.ndarray): 去重并排序后的x值数组。
    y_sorted (numpy.ndarray): 对应去重并排序后的y值数组。
    """
    # 将输入转换为numpy数组
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 输入验证
    if len(x) != len(y):
        raise ValueError("x和y的长度必须相同")

    # 去重
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]

    # 重新排序
    sorted_idx = np.argsort(x_unique)
    x_sorted = x_unique[sorted_idx]
    y_sorted = y_unique[sorted_idx]

    # 生成插值点
    # xInterp = np.linspace(x_sorted.min(), x_sorted.max(), num_points)
    xInterp = np.arange(x_sorted.min(), x_sorted.max()+1)

    # 使用pchip插值
    pchip = PchipInterpolator(x_sorted, y_sorted)
    pchipInterp = pchip(xInterp)

    return xInterp, pchipInterp, x_sorted, y_sorted


def downsample_time_series(data, label_x, label_y, window_size=3000, label_sample_mode='mean'):
    """
    对时间序列数据进行降采样，使用窗口大小对数据进行分组，并将每个窗口内的数据累加到一列中。
    参数:
    label_x (array-like): 标签数组，必须与 data 的长度一致。
    label_y (array-like): 标签数组，必须与 data 的长度一致。
    data (array-like): 数据数组，可以是1D或2D。
    window_size (int): 窗口大小，表示每个窗口的采样点数。
    label_sample_mode (str): 'mean' 表示对标签取均值，'sample' 表示直接对标签抽样取值，默认为'mean'。
    返回:
    downsampled_data (numpy.ndarray): 降采样后的数据数组。
    downsampled_label (numpy.ndarray): 降采样后的标签数组。
    """
    # 将输入转换为NumPy数组
    data = np.asarray(data)
    label_x = np.asarray(label_x)
    label_y = np.asarray(label_y)

    # 输入验证
    if len(label_x) != data.shape[1]:
        raise ValueError("标签与神经数据长度必须一致")
    if len(label_y) != data.shape[1]:
        raise ValueError("标签与神经数据长度必须一致")

    # 计算窗口数量
    num_windows = data.shape[1] // window_size

    # 降采样数据
    downsampled_data = data[:, :num_windows * window_size].reshape(-1, num_windows, window_size).sum(axis=2)

    if label_sample_mode == 'mean':
        # 降采样标签，取均值
        downsampled_label_x = label_x[:num_windows * window_size].reshape(-1, window_size).mean(axis=1)
        downsampled_label_y = label_y[:num_windows * window_size].reshape(-1, window_size).mean(axis=1)
    elif label_sample_mode == 'sample':
        # 直接对标签抽样取值，这里取的是窗口开始位置的标签值
        downsampled_label_x = label_x[:num_windows * window_size].reshape(-1, window_size)[:, -1]
        downsampled_label_y = label_y[:num_windows * window_size].reshape(-1, window_size)[:, -1]
    else:
        raise ValueError(f"不支持的标签抽样模式：{label_sample_mode}")

    # 拼接标签
    downsampled_label = np.stack((downsampled_label_x, downsampled_label_y), axis=0)

    return downsampled_data, downsampled_label

def create_windowed_dataset(data,label,trial_labels, window_size, stride):
    """
    根据时间窗划分数据集。
    :param window_size: 时间窗长度。
    :param stride: 滑动步长。
    :return: features_train, targets_train, features_test, targets_test。
    """
    # 加载数据
    feature_matrixs = data
    labels = label

    # 存储数据
    all_features=[]
    all_labels=[]
    all_trial_label_list=[]
    def window_slide(trial_data,label,i):
        features = []
        targets = []
        trial_label=[]
        for start in range(0, trial_data.shape[0] - window_size + 1, stride):
            end = start + window_size
            window_features = trial_data[start:end]
            window_label = label[start:end]
            if end <= trial_data.shape[0]:
                features.append(window_features)
                targets.append(window_label)
                trial_label.append(trial_labels[i])

        return np.array(features, dtype=np.int8), np.array(targets, dtype=np.float32), np.array(trial_label, dtype=np.int8)

    for i in range(len(feature_matrixs)):

        if feature_matrixs[i].shape[0]<window_size:
            pass
        else:
            trial_feature,trial_target,trial_label_final=window_slide(feature_matrixs[i],labels[i],i)
            all_features.append(trial_feature)
            all_labels.append(trial_target)
            all_trial_label_list.append(trial_label_final)

    all_features_np=np.concatenate(all_features,axis=0)
    all_labels_np=np.concatenate(all_labels,axis=0)
    all_trial_labels_np=np.concatenate(all_trial_label_list,axis=0)


    return all_features_np,all_labels_np,all_trial_labels_np


def prepare_dataloader(features_train, targets_train, features_test, targets_test, batch_size=16):
    """
    将特征和标签转换为 PyTorch DataLoader。

    :param features: 特征数组，形状为 (样本数, window_size, 特征数)。
    :param targets: 标签数组，形状为 (样本数, 2)。
    :param batch_size: DataLoader 的批量大小。
    :param test_size: 测试集比例。
    :return: train_loader, test_loader。
    """

    # 转为 PyTorch 张量
    features_train = torch.tensor(features_train, dtype=torch.float32)
    targets_train = torch.tensor(targets_train, dtype=torch.float32)

    features_test = torch.tensor(features_test, dtype=torch.float32)
    targets_test = torch.tensor(targets_test, dtype=torch.float32)

    # 创建 DataLoader
    train_loader = DataLoader(TensorDataset(features_train, targets_train), batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(TensorDataset(features_test, targets_test), batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader



if __name__ == "__main__":

    # data (num, seq_len, channel),  label(num, feature_dim, seq_len)

    # 打开MAT文件
    # date="0320"'0402','0403','0408','0409','0410','0411','0414','0415','0416','0417','0418','0421','0422',
    for date in ['0401','0402','0403','0408','0409','0410','0411','0414','0415','0416','0417','0418','0421','0422','0423','0424','0425','0427','0428','0429','0430']:
        window=100
        bin_size=20000
        step=5

        file_path= "../20250324_from_lx/"+date+"_v7.3.mat"
        save_path = f"../data/" + date + f"_window_{window}_step_{step}_trial_down_{bin_size}_label_mean_position.pkl"
        print('Processing date :',date)
        # 获取试验状态
        trial_status=dereference_h5_refs(file_path,"data/Status")
        # 获取目标点label
        trial_label_list=dereference_h5_refs(file_path,"data/TargetPtIndex")
        # 获取行为数据
        behaviour_list=dereference_h5_refs(file_path,"data/rawBehavior")
        # 获取神经数据
        spike_time_list=dereference_h5_spike(file_path,"data/cts")

        data_list=[]
        label_list=[]

        all_features_train = []
        all_targets_train = []
        all_features_test = []
        all_targets_test = []

        all_trial_label_train=[]
        all_trial_label_test=[]
        trial_label_list_success=[]

        for trial_idx in range(len(trial_status)):
            if trial_status[trial_idx]["data"]=="Success":

                trial_label=trial_label_list[trial_idx]

                trial_label_list_success.append(int(trial_label['data'][0]))

                trial_index = trial_label['data']

                behaviour=behaviour_list[trial_idx]
                spike_time=spike_time_list[trial_idx]

                # TRIAL时间戳
                trial_behaviour_time=behaviour["data"]["nowTimeStamp"]

                # 起止时间
                trial_start_time=trial_behaviour_time[0]
                trial_end_time = trial_behaviour_time[-1]
                time_interval=int(trial_end_time-trial_start_time)

                # 标签信息
                x_label=behaviour["data"]["analogX"]
                y_label=behaviour["data"]["analogY"]

                # x_label = behaviour["data"]["cursorX"]
                # y_label = behaviour["data"]["cursorY"]

                # 重构脉冲矩阵 （1us）
                spike_matrix=reconstruct_spike_matrix(spike_time,trial_start_time,trial_end_time,time_interval)

                # 根据脉冲矩阵对行为标签插值（分段插值）
                _, pchipInterp_x, _, _ = pchip_interpolation(trial_behaviour_time, x_label)
                _, pchipInterp_y, _, _ = pchip_interpolation(trial_behaviour_time, y_label)


                # 降采样,最小时间戳为1us
                downsampled_data, downsampled_label=downsample_time_series(spike_matrix,pchipInterp_x,pchipInterp_y,window_size=bin_size,label_sample_mode='mean')

                window_length = 15 # 窗口长度，必须是奇数
                polyorder = 5  # 多项式阶数
                filtered_voltage_x = signal.savgol_filter(downsampled_label[0,:], window_length, polyorder)
                filtered_voltage_y = signal.savgol_filter(downsampled_label[1,:], window_length, polyorder)


                # 绘制结果
                # plt.figure(figsize=(10, 6))
                # plt.subplot(2, 1, 1)
                # plt.plot(downsampled_label[0,:], label='Interpolated X')
                # plt.plot(filtered_voltage_x, label='Filtered X')
                # plt.legend()
                # plt.subplot(2, 1, 2)
                # plt.plot(downsampled_label[1,:], label='Interpolated Y')
                # plt.plot(filtered_voltage_y, label='Filtered Y')
                # plt.legend()
                # plt.show()

                downsampled_label = np.stack((filtered_voltage_x, filtered_voltage_y), axis=0)
                print(f"After downsample : Data shape {downsampled_data.shape}, Label shape {downsampled_label.shape}")

                data_list.append(downsampled_data.T)
                label_list.append(downsampled_label.T)

        # 滑窗bin，并且按前后顺序划分训练，测试集
        train_ratio = 0.8
        num_trials=len(data_list)
        num_train = int(num_trials * train_ratio)
        num_test = num_trials - num_train
        # 训练集
        features_train, labels_train,trial_labels_train = create_windowed_dataset(data_list[:num_train], label_list[:num_train],trial_label_list_success[:num_train], window_size=window, stride=step)
        print(f"TRAIN Window data_shape:{features_train.shape}, labels_shape:{labels_train.shape}")

        features_test, labels_test,trial_labels_test= create_windowed_dataset(data_list[num_train:], label_list[num_train:],trial_label_list_success[num_train:],window_size=window, stride=step)
        print(f"TEST Window data_shape:{features_test.shape}, labels_shape:{labels_test.shape}")

        # 按索引划分数据
        all_features_train.append(features_train)
        all_targets_train.append(labels_train)
        all_trial_label_train.append(trial_labels_train)

        all_features_test.append(features_test)
        all_targets_test.append(labels_test)
        all_trial_label_test.append(trial_labels_test)

        # 将所有 session 的数据拼接成一个大的数据集
        all_features_train = np.concatenate(all_features_train, axis=0)  # 沿样本维度拼接
        all_targets_train = np.concatenate(all_targets_train, axis=0)  # 沿样本维度拼接
        all_trial_label_train = np.concatenate(all_trial_label_train, axis=0)  # 沿样本维度拼接

        all_features_test = np.concatenate(all_features_test, axis=0)
        all_targets_test = np.concatenate(all_targets_test, axis=0)
        all_trial_label_test = np.concatenate(all_trial_label_test, axis=0)

        print(f"#############训练总样本数: {all_features_train.shape}, 标签维度：{all_targets_train.shape},测试总样本数: {all_features_test.shape}, 标签维度：{all_targets_test.shape}#################")

        with open(save_path, 'wb') as f:
            pickle.dump({'features_train': all_features_train, 'targets_train': all_targets_train,'trial_label_train':all_trial_label_train,'features_test': all_features_test, 'targets_test': all_targets_test,'trial_label_test':all_trial_label_test}, f)

        print(f'2025{date} split done!!!!')
















