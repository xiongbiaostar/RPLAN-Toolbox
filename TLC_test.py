import pickle

import matplotlib.pyplot as plt
import numpy as np

from rplan.floorplan import Floorplan
from rplan.align import align_fp_gt
from rplan.decorate import get_dw
from rplan.measure import compute_tf
from rplan.plot import get_figure, get_axes, plot_category, plot_boundary, plot_graph, plot_fp, plot_tf, plot_fp_TCL
from tqdm import tqdm


type_order = [1,2,3,4,1,2,2,2,2,5,1,6,1,10,7,8,9,10]

def point_inside(point, box):
    # 检查点是否在边界框内
    x_min, x_max = min(box[0], box[2]), max(box[0], box[2])
    y_min, y_max = min(box[1], box[3]), max(box[1], box[3])
    return x_min <= int(point[0]) <= x_max and y_min <= int(point[1]) <= y_max

def room_inside(room, box):
    # 检查房间的所有角点是否都在边界框内
    return all(point_inside(point, box) for point in room)
def assign_types(room_boundaries, types_bboxes, types):
    room_types = np.full(len(room_boundaries), 16, dtype=np.int32)
    for i, type_bbox in enumerate(types_bboxes):
        if type_order[types[i]] > 6:
            continue
        for j, room in enumerate(room_boundaries):
            if room_inside(room, type_bbox):
                room_types[j] = type_order[types[i]]
    return room_types



# RPLAN_DIR = './data'
# file_path = f'{RPLAN_DIR}/0.png'
# fp = Floorplan(file_path)
# data = fp.to_dict()
#
# boxes_aligned, order, room_boundaries = align_fp_gt(data['boundary'],data['boxes'],data['types'],data['edges'])
# data['boxes_aligned'] = boxes_aligned
# data['order'] = order
# data['room_boundaries'] = room_boundaries
#
# doors,windows = get_dw(data)
# data['doors'] = doors
# data['windows'] = windows
#
# print(data['boxes_aligned'][order])
# print(data['types'][order])
#
# fig = get_figure([512,512])
# plot_fp(data['boundary'], data['boxes_aligned'][order], data['types'][order],data['doors'],data['windows'],ax=get_axes(fig=fig,rect=[0.25,0.25,0.5,0.5]))
# fig.canvas.draw()
# fig.canvas.print_figure('./output/plot.png')
#
# x,y = compute_tf(data['boundary'])
# plot_tf(x,y)
# plt.savefig('./output/tf.png')
# plt.close()
#
#
# boxes_types = assign_types(room_boundaries, data['boxes_aligned'][order], data['types'][order])
#
# print(boxes_types)

def normalize(data, boundary,target_min=17, target_max=110):
    boundary = np.array(boundary)
    max_data = boundary.max()
    min_data = boundary.min()
    shape = 2
    min_data = np.array([min_data] * shape)
    max_data = np.array([max_data] * shape)
    data = np.array(data, dtype=np.float32)
    normalized_data = (data - min_data) / (max_data - min_data)

    scaled_data = normalized_data * (target_max - target_min) + target_min
    scaled_data = np.array(scaled_data, dtype=np.float32)
    scaled_data = np.clip(scaled_data, a_min=target_min, a_max=target_max)
    scaled_data = np.round(scaled_data)
    # 转换为整数
    scaled_data = scaled_data.astype('int32')
    return scaled_data

import numpy as np
from math import atan2, degrees

def compute_direction(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = atan2(dy, dx)
    angle_deg = (degrees(angle) + 360) % 360
    direction = int(round(angle_deg / 90)) % 4
    return direction

def convert_boundary(boundary_quan):
    n = len(boundary_quan)
    result = []

    for i in range(n):
        x, y = boundary_quan[i]
        next_point = boundary_quan[(i + 1) % n]  # 循环连接
        direction = compute_direction(boundary_quan[i], next_point)
        flag = 1 if i < 2 else 0
        result.append([x, y, direction, flag])

    return np.array(result, dtype=np.int32)


import numpy as np

def ensure_clockwise(points, boundary):
    """
    确保多边形点顺序与 boundary 的顺序相同。

    参数:
        points (np.ndarray or list): [n, 2] 的点列表
        boundary (np.ndarray or list): [m, 2] 的点列表，表示边界多边形

    返回:
        np.ndarray: 与 boundary 顺序相同的点
    """
    points = np.array(points)
    boundary = np.array(boundary)

    # 计算 boundary 的方向
    x = boundary[:, 0]
    y = boundary[:, 1]
    boundary_area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    boundary_area += x[-1] * y[0] - x[0] * y[-1]  # 闭合首尾

    # 计算 points 的方向
    x = points[:, 0]
    y = points[:, 1]
    points_area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    points_area += x[-1] * y[0] - x[0] * y[-1]  # 闭合首尾

    # 如果 points 的方向与 boundary 的方向不同，则翻转 points 的顺序
    if (boundary_area > 0 and points_area < 0) or (boundary_area < 0 and points_area > 0):
        points = points[::-1]

    return points


# RPLAN_DIR = './data'
# file_path = f'{RPLAN_DIR}/0.png'
# fp = Floorplan(file_path)
# a = fp.category
# data = fp.to_dict()
#
# boxes_aligned, order, room_boundaries = align_fp_gt(data['boundary'],data['boxes'],data['types'],data['edges'])
# data['boxes_aligned'] = boxes_aligned
# data['order'] = order
# data['room_boundaries'] = room_boundaries
#
# doors,windows = get_dw(data)
# data['doors'] = doors
# data['windows'] = windows
#
# fig = get_figure([512, 512])
# plot_fp(data['boundary'], data['boxes_aligned'], data['types'], data['doors'], data['windows'],
#         ax=get_axes(fig=fig, rect=[0, 0, 1, 1]))
# fig.canvas.draw()
# fig.canvas.print_figure('./output/plot.png')
#
# x, y = compute_tf(data['boundary'])
# plot_tf(x, y)
# plt.savefig('./output/tf.png')
# plt.close()
#
with open('./TCL/gt.pkl', 'rb') as f:  # profile/train.py
    room_data = pickle.load(f)

last_uid = ''
index = 0
for room in tqdm(room_data, desc="Processing rooms", unit="room"):

    try:
        uid = room['uid']
        # if uid != '77243' or uid != '72682':
        #     continue
        if uid == last_uid:
            index += 1
        else:
            index = 0
            last_uid = uid
        room_all = room['param']
        skip_room = False
        for room_i in room_all:
            if len(room_i) < 5:
                skip_room = True  # 设置标志变量
                break  # 跳出内层循环

        if skip_room:
            continue
        boundary = np.array(room_all[0][1:])
        boundary_quan =  normalize(boundary, boundary)
        room_boundaries = np.array([row[1:] for row in room_all[1:]])
        #types = np.array([row[0][0] for row in room_all[1:]]) - 1             #pred
        types = np.array([row[0][0] for row in room_all[1:]]) - 2               #Gt
        room_boundaries_quan = []
        for room_i in room_boundaries:
            room_boundaries_quan.append(normalize(room_i, boundary))
        room_boundaries_quan = np.array(room_boundaries_quan)

        room_align = []
        for room_bq in room_boundaries_quan:
            room_align.append(ensure_clockwise(room_bq, boundary_quan))

        room_boundaries_quan = np.array(room_align)


        boundary_quan = convert_boundary(boundary_quan)

        boxes = []
        # 处理每个子房间的 bounding box
        for room_quan in room_boundaries_quan:
            room_quan = np.array(room_quan)
            xmin, ymin = np.min(room_quan, axis=0)
            xmax, ymax = np.max(room_quan, axis=0)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = np.array(boxes)

        # 获取排序索引
        # sorted_indices = np.argsort(types)
        #
        # # 根据排序索引调整 types 和 room_boundaries_quan 的顺序
        # types = types[sorted_indices]
        # room_boundaries_quan = room_boundaries_quan[sorted_indices]
        # boxes = boxes[sorted_indices]

        data ={'name' : uid,
               'types' : types,
               'boxes_aligned' : boxes,
               'room_boundaries' : room_boundaries_quan,
               'boundary' : boundary_quan,
                }

        doors,windows = get_dw(data)
        data['doors'] = doors
        data['windows'] = windows

        fig = get_figure([512, 512])
        ax = get_axes(fig=fig, rect=[0, 0, 1, 1])  # 先拿到 ax
        #plot_boundary(data['boundary'], ax=ax)
        plot_fp_TCL(data['boundary'], data['room_boundaries'], data['types'],data['doors'],data['windows'],ax=ax)

        # 自动调整坐标轴
        x_vals = data['boundary'][:, 0]
        y_vals = data['boundary'][:, 1]

        xmin, xmax = x_vals.min(), x_vals.max()
        ymin, ymax = y_vals.min(), y_vals.max()
        # ax.set_xlim(xmin - 6, xmax + 6)
        # ax.set_ylim(ymin - 6, ymax + 6)

        ax.set_xlim(xmin , xmax )
        ax.set_ylim(ymin , ymax)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        fig.canvas.draw()
        # if index == 0:
        #     fig.canvas.print_figure(f"./output_infrontdoor_1000epoch_gt/{uid}.png")
        # else:
        #     fig.canvas.print_figure(f"./output_infrontdoor_1000epoch/{uid}.png")
        fig.canvas.print_figure(f"./test/{uid}_{0}.png")
    except Exception as e:
        print(e)


    #plot_boundary(data['boundary'],ax=get_axes(fig=fig,rect=[0,0.5,0.5,0.5]))
    #ax = plot_category(category=data['types'],ax=get_axes(fig=fig,rect=[0.5,0.5,0.5,0.5]))
    #plot_graph(data['boundary'],data['boxes'],data['types'],data['edges'],ax=ax)
    #plot_fp(data['boundary'], data['boxes_aligned'][order], data['types'][order],ax=get_axes(fig=fig,rect=[0,0,0.5,0.5]))