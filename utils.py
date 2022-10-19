from logging.config import valid_ident
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy import stats

def get_dist(lat1, lon1, lat2, lon2):
    R = 6371e3
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

def generate_series(data, y, window_size, pred_size, date=False):
    '''
    Data: N*T*F
    y: N*T
    '''
    series = []
    targets = []
    idx = window_size
    while idx + pred_size < data.shape[1]:
        if date:
            series.append(data[:, idx-1, :])
        else:
            series.append(np.sum(data[:, idx-window_size:idx, :], axis=1))
        targets.append(np.sum(y[:, idx:idx+pred_size], axis=1))
        idx += pred_size
    return np.array(series).transpose(1,0,2), np.array(targets).transpose(1,0)

def temporal_split(x, y, static, mats, val_ratio, test_ratio, norm='z-score', norm_mat = True):
    seq_len = x.shape[1]
    test_len = int(seq_len * test_ratio)
    val_len = int(seq_len * val_ratio)
    train_len = seq_len - val_len - test_len
    
    shuffle_idx = np.arange(x.shape[0])
    np.random.shuffle(shuffle_idx)
    x = x[shuffle_idx, :, :]
    y = y[shuffle_idx, :]
    static = static[shuffle_idx, :]
    for i in range(len(mats)):
        mats[i] = mats[i][shuffle_idx, :][:, shuffle_idx]
    
    idx = np.arange(x.shape[1])
    train_x = x[:, :train_len, :]
    train_y = y[:, :train_len]
    train_idx = idx[:train_len]
    val_x = x[:, train_len:train_len+val_len, :]
    val_y = y[:, train_len:train_len+val_len]
    val_idx = idx[train_len:train_len+val_len]
    test_x = x[:, train_len+val_len:, :]
    test_y = y[:, train_len+val_len:]
    test_idx = idx[train_len+val_len:]
    
    normalize_dict = {}
    for i in range(x.shape[2]):
        normalize_dict[i] = [np.mean(train_x[:, :, i]), np.std(train_x[:, :, i]), np.min(train_x[:, :, i]), np.max(train_x[:, :, i])]
    normalize_dict['y'] = [np.mean(train_y[:, :]), np.std(train_y[:, :]), np.min(train_y[:, :]), np.max(train_y[:, :])]
    normalize_dict['static'] = [np.mean(static, axis=0), np.std(static, axis=0), np.min(static, axis=0), np.max(static, axis=0)]
    for i in range(len(mats)):
        normalize_dict['mat_%d'%i] = [np.mean(mats[i]), np.std(mats[i]), np.min(mats[i]), np.max(mats[i])]

    for i in range(train_x.shape[2]):
        if (normalize_dict[i][3] - normalize_dict[i][2]) == 0:
            train_x[:, :, i] = 0
            val_x[:, :, i] = 0
            test_x[:, :, i] = 0
        else:
            train_x[:, :, i] = (train_x[:, :, i] - normalize_dict[i][2]) / (normalize_dict[i][3] - normalize_dict[i][2])
            val_x[:, :, i] = (val_x[:, :, i] - normalize_dict[i][2]) / (normalize_dict[i][3] - normalize_dict[i][2])
            test_x[:, :, i] = (test_x[:, :, i] - normalize_dict[i][2]) / (normalize_dict[i][3] - normalize_dict[i][2])
    train_y = (train_y - normalize_dict['y'][0]) / normalize_dict['y'][1]
    val_y = (val_y - normalize_dict['y'][0]) / normalize_dict['y'][1]
    test_y = (test_y - normalize_dict['y'][0]) / normalize_dict['y'][1]
    static = (static - normalize_dict['static'][2]) / (normalize_dict['static'][3] - normalize_dict['static'][2])
    if norm_mat:
        for i in range(len(mats)):
            mats[i] = (mats[i] - normalize_dict['mat_%d'%i][2]) / (normalize_dict['mat_%d'%i][3] - normalize_dict['mat_%d'%i][2])

    return train_x, val_x, test_x, train_y, val_y, test_y, train_idx, val_idx, test_idx, static, mats, normalize_dict, shuffle_idx

def mse(y_true, y_pred, std=False):
    if std:
        return np.mean((y_true - y_pred)**2, axis=1).mean(), np.mean((y_true - y_pred)**2, axis=1).std()
    else:
        return np.mean((y_true - y_pred)**2, axis=1).mean()

def mae(y_true, y_pred, std=False):
    if std:
        return np.mean(np.abs(y_true - y_pred), axis=1).mean(), np.mean(np.abs(y_true - y_pred), axis=1).std()
    else:
        return np.mean(np.abs(y_true - y_pred), axis=1).mean()

def r2(y_true, y_pred, std=False):
    return r2_score(y_true, y_pred)

def ccc(y_true, y_pred, std=False):
    return stats.pearsonr(y_true.flatten(), y_pred.flatten())[0]