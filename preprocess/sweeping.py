import numpy as np
from dataload import load_data
import pylab as plt
from utils import normalize
from utils import quantize_to_6bit
from utils import save
# 这里是扫线法, 
# 优点是可以很好彻底分离外边框与房屋边框, 
# 缺点是会丢失线段顺序信息, 而且分离外边框存在作弊嫌疑

# 切分函数
def get_line(room_boundaries):
    ret = []
    for i in room_boundaries:
        for j in i:
            ret.append(j)
    ret = np.array(ret)
    x = ret.T[0]
    y = ret.T[1]
    set_x = set(x)
    set_y = set(y)
    line = []
    for _ in set_x:
        index = np.where(x == _)
        first = True
        last_point = []
        this_point = []
        for i, j in zip(x[index], np.sort(y[index])):
            if first:
                this_point = [i, j]
                first = False
                continue
            else:
                last_point = this_point
                this_point = [i, j]
                if last_point != this_point:

                    line.append([this_point, last_point])
    for _ in set_y:
        index = np.where(y == _)
        first = True
        last_point = []
        this_point = []
        for i, j in zip(np.sort(x[index]), y[index]):
            if first:
                this_point = [i, j]
                first = False
                continue
            else:
                last_point = this_point
                this_point = [i, j]
                if last_point != this_point:

                    line.append([this_point, last_point])

    return np.array(line)

from shapely import geometry
def split_line(line, boundary, room_boundary):
    poly_boundary = geometry.Polygon(boundary)
    boundary_line = []
    room_boundaries_line = []

    for _ in line:
        # 创建点对象
        line_ = geometry.LineString([(_.T[0][0], _.T[1][0]), (_.T[0][1], _.T[1][1])])
        is_inside = poly_boundary.exterior.contains(line_)
        if is_inside:
            boundary_line.append([_[0], _[1]])
        else:
            # 解决学长之前提到的bug, 扫线的不一定是最终的房间边框
            flag = True
            if poly_boundary.contains(line_):
                for j in room_boundary:
                    room_poly = geometry.Polygon(j)
                    if room_poly.contains(line_):
                        flag = False
                if flag:
                    room_boundaries_line.append([_[0], _[1]])
    return np.array(boundary_line), np.array(room_boundaries_line)

def perprocess(data:dict):
    ret_param = []
    ret_seq_mask = []
    ret_ignore_mask = []

    for i in list(data.keys()):
        try: # 这里用try catch 主要是应对room_boundaries部分存在nan值的问题
            # 这里就是一个线段line -> (x1, y1, x2, y2)
            param = np.zeros((80, 4))
            lines = get_line(data[i]['room_boundaries'])
            boundary_line, room_boundaries_line = split_line(lines, data[i]['boundary'][:, :2])

            boundary_line = normalize(boundary_line, data[i]['boundary'])
            room_boundaries_line = normalize(room_boundaries_line, data[i]['boundary'])

            boundary_line_shape = boundary_line.shape[0]
            room_boundaries_line_shape = room_boundaries_line.shape[0]
            param[:boundary_line_shape, :] = boundary_line.reshape(boundary_line_shape, -1)
            param[boundary_line_shape:boundary_line_shape + room_boundaries_line_shape, :] = room_boundaries_line.reshape(room_boundaries_line_shape, -1)
            param = quantize_to_6bit(param)
            seq_mask = np.full((80), False, dtype=bool)
            seq_mask[:lines.shape[0]] = True

            ignore_mask = np.full((80, 4), False, dtype=bool)
            ignore_mask[boundary_line_shape:boundary_line_shape + room_boundaries_line_shape, :] = True
            
            ret_param.append(param)
            ret_seq_mask.append(seq_mask)
            ret_ignore_mask.append(ignore_mask)
        except:
            print(i)

    save(ret_param, ret_seq_mask, ret_ignore_mask)

# 示例函数
if __name__ == '__main__':
    data = load_data('data/')
    lines = get_line(data[0]['room_boundaries'])
    boundary_line, room_boundaries_line = split_line(lines, data[0]['boundary'][:, :2], data[0]['room_boundaries'])
    boundary_line = normalize(boundary_line, data[0]['boundary'])
    room_boundaries_line = normalize(room_boundaries_line, data[0]['boundary'])
    print(lines.shape, boundary_line.shape, room_boundaries_line.shape)
    t = 0
    for _ in boundary_line:
        if _.T[0].min() ==  _.T[0].max():
            y = [_.T[1].min() + t , _.T[1].max() - t]
            x = [_.T[0].min(), _.T[0].max()]
        else:
            x = [_.T[0].min() + t, _.T[0].max() - t]
            y = [_.T[1].min(), _.T[1].max()]
        plt.plot(x, y)
    plt.axis('equal')
    plt.show()
    for _ in room_boundaries_line:
        if _.T[0].min() ==  _.T[0].max():
            y = [_.T[1].min() + t , _.T[1].max() - t]
            x = [_.T[0].min(), _.T[0].max()]
        else:
            x = [_.T[0].min() + t, _.T[0].max() - t]
            y = [_.T[1].min(), _.T[1].max()]
        plt.plot(x, y)
    plt.axis('equal')
    plt.show()