import numpy as np
import pickle
def quantize_to_6bit(normalized_data):
    # 将数据量化到 [0, 63] 的整数范围
    quantized_data = np.round(normalized_data * 63).astype(np.uint8)
    return quantized_data

def normalize(line, boundary):
    # 归一化
    w_max, w_min, h_max, h_min = boundary[:, 0].max(), boundary[:, 0].min(),boundary[:, 1].max(), boundary[:, 1].min()
    wh_max = max(w_max, h_max)
    wh_min = min(w_min, h_min)

    return (line - wh_min) / (wh_max - wh_min)

def rotate(line, theta):
    # 选择, 传入线段x1, y1, x2, y2
    ret = []
    for _ in line:
        line1, line2 = _
        x1, y1 = line1
        x2, y2 = line2
        x_f = lambda x, y: x * np.cos(theta) - y * np.sin(theta)
        y_f = lambda x, y: x * np.sin(theta) + y * np.cos(theta)
        
        x1_new = x_f(x1, y1)
        y1_new = y_f(x1, y1)
        x2_new = x_f(x2, y2)
        y2_new = y_f(x2, y2)
        
        ret.append([[x1_new, y1_new], [x2_new, y2_new]])
    return np.array(ret)

def save(ret_param, ret_seq_mask, ret_ignore_mask):
    # 保存数据
    param = np.stack(ret_param, axis=0)
    seq_mask = np.stack(ret_seq_mask, axis=0)
    ignore_mask = np.stack(ret_ignore_mask, axis = 0)

    train_param = np.delete(param[:int(param.shape[0]* 0.7)], np.arange(89 * 256, 90 * 256), axis = 0)
    train_seq_mask = np.delete(seq_mask[:int(seq_mask.shape[0] * 0.7)], np.arange(89 * 256, 90 * 256), axis = 0)
    train_ignore_mask = np.delete(ignore_mask[:int(ignore_mask.shape[0] * 0.7)], np.arange(89 * 256, 90 * 256), axis = 0)

    val_param = param[int(param.shape[0]* 0.7):int(param.shape[0]* 0.9)]
    val_seq_mask = seq_mask[int(seq_mask.shape[0] * 0.7):int(param.shape[0]* 0.9)]
    val_ignore_mask = ignore_mask[int(ignore_mask.shape[0] * 0.7):int(param.shape[0]* 0.9)]

    test_param = param[int(param.shape[0]* 0.9):]
    test_seq_mask = seq_mask[int(seq_mask.shape[0] * 0.9):]
    test_ignore_mask = ignore_mask[int(ignore_mask.shape[0] * 0.9):]

    with open('train.pkl', 'wb') as f:
        pickle.dump({'param': train_param, 'seq_mask': train_seq_mask, 'ignore_mask': train_ignore_mask}, f)

    with open('test.pkl', 'wb') as f:
        pickle.dump({'param': test_param, 'seq_mask': test_seq_mask, 'ignore_mask': test_ignore_mask}, f)

    with open('val.pkl', 'wb') as f:
        pickle.dump({'param': val_param, 'seq_mask': val_seq_mask, 'ignore_mask': val_ignore_mask}, f)