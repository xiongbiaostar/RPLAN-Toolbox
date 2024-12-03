# 这个是最新的表示方法
# [x1, y1], [x2, y2], [0, 0]
from dataload import load_data
import numpy as np
import pylab as plt
from utils import quantize_to_6bit
from utils import normalize
from utils import save

def draw_lines(param, seq_mask, ignore_mask):
    # 根据 param, seq_mask, ignore_mask 绘制图像
    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(0, param.shape[0], 3): 
        if seq_mask[i] and not ignore_mask[i, 0]:
            x1, y1 = param[i]
            x2, y2 = param[i + 1]

            ax.plot([x1, x2], [y1, y2], 'o-', color='blue', linewidth=2)
    plt.axis('equal')
    plt.title('boundary')
    plt.show()
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(0, param.shape[0], 3): 
        if seq_mask[i] and ignore_mask[i, 0]:
            x1, y1 = param[i]
            x2, y2 = param[i + 1]

            ax.plot([x1, x2], [y1, y2], 'o-', color='blue', linewidth=2)
    plt.axis('equal')
    plt.title('room')
    plt.show()

if __name__ == '__main__':
    data = load_data('data/')
    ret_param = []
    ret_seq_mask = []
    ret_ignore_mask = []
    for _ in data.keys():
        # try:
            param = np.zeros((450, 2))
            seq_mask = np.full((450), False, dtype=bool)
            ignore_mask = np.full((450, 2), False, dtype=bool)
            boundary = data[_]['boundary'][:, :2]
            lines_array = []

            for i in range(len(boundary)):        
                start_point = boundary[i]
                end_point =  boundary[(i + 1) % len(boundary)]

                point1 = quantize_to_6bit(
                        normalize(
                            [start_point[0], start_point[1]], 
                            boundary
                        )
                    )

                point2 = quantize_to_6bit(
                        normalize(
                            [end_point[0], end_point[1]], 
                            boundary
                        )
                    )

                line_data = np.array([point1, 
                                        point2, 
                                        [0, 0]])
                    
                lines_array.append(line_data)

            boundary_len = len(lines_array)
            flag = True
            for i in data[_]['room_boundaries']:
                # 防止nan
                if np.any(np.isnan(np.array(i).astype(float))):
                    print(_)
                    flag = False
                    break
                for j in range(len(i)):
                    start_point = i[j]
                    end_point = i[(j + 1) % len(i)]
                    
                    point1 = quantize_to_6bit(
                        normalize(
                            [start_point[0], start_point[1]], 
                            boundary
                        )
                    )

                    point2 = quantize_to_6bit(
                        normalize(
                            [end_point[0], end_point[1]], 
                            boundary
                        )
                    )

                    line_data = np.array([point1, 
                                        point2, 
                                        [0, 0]])
                    
                    lines_array.append(line_data)
            if not flag: 
                continue
            lines_array = np.array(lines_array).reshape(-1, 2)
            param[:lines_array.shape[0]] = lines_array
            seq_mask[np.arange(0, lines_array.shape[0], 3)] = True
            seq_mask[np.arange(1, lines_array.shape[0], 3)] = True
            ignore_mask[np.arange(boundary_len * 3, lines_array.shape[0], 3)] = True
            ignore_mask[np.arange(boundary_len * 3 + 1, lines_array.shape[0], 3)] = True
            
            ret_param.append(param)
            ret_seq_mask.append(seq_mask)
            ret_ignore_mask.append(ignore_mask)
            save(ret_param, ret_seq_mask, ret_ignore_mask)
        # except:
        #     print(_)

    draw_lines(ret_param[0], ret_seq_mask[0], ret_ignore_mask[0])
