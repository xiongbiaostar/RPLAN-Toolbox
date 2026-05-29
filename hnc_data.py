import pickle
import warnings

import numpy as np
import os

from shapely import Polygon
from tqdm import tqdm

def boundary_rotation(data, boundary, theta):
    # 计算边界框的最小和最大值
    points_x, points_y = boundary.T

    # 分别求 x 和 y 的最小值和最大值
    min_x = np.min(points_x)
    max_x = np.max(points_x)
    min_y = np.min(points_y)
    max_y = np.max(points_y)

    # 计算边界框的中心点
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center = np.array([center_x, center_y])

    # 平移点集到原点
    translated_data = data - center

    # 旋转点集
    points_x, points_y = translated_data.T
    x = points_x * np.cos(theta) - points_y * np.sin(theta)
    y = points_x * np.sin(theta) + points_y * np.cos(theta)
    rotated_data = np.stack([x, y]).T

    # 平移回原来的位置
    rotated_data += center

    return rotated_data

def normalize(data, boundary, type='boundary'):
    max_data = boundary.max()
    min_data = boundary.min()
    if type == 'boundary':
        shape = 2
    else:
        shape = 4
    min_data = np.array([min_data] * shape)
    max_data = np.array([max_data] * shape)
    data = (data - min_data) / (max_data - min_data)

    quantized_data = data * 63
    quantized_data = np.array(quantized_data, dtype=np.float32)
    quantized_data = np.clip(quantized_data, a_min=0, a_max=63)  # 限制值的范围
    quantized_data = np.round(np.array(quantized_data))
    # 转换为整数
    quantized_data = quantized_data.astype('int32')
    return quantized_data


def gen_boundary_data(ret, if_rotation=False):
    param = []
    if if_rotation:
        theta_list = [0, np.pi, np.pi / 2, -np.pi / 2]
    else:
        theta_list = [0]

    # 使用 tqdm 来显示进度条
    for i, key in enumerate(tqdm(list(ret.keys()), desc='Generating boundary data')):

        index = i
        for theta in theta_list:

            t2 = 0
            boundary = ret[key]['boundary'][:, :2]
            room_boundaries = ret[key]['room_boundaries']
            boundary_non_norm = boundary_rotation(boundary, theta)
            boundary = normalize(boundary_non_norm, boundary_non_norm, 'boundary')
            param.append(
                {'param': boundary, 'uid': f'{index:06}_{t2}'}
            )


            for room_boundary in room_boundaries:
                room = np.array(room_boundary, dtype=np.float32)
                if np.isnan(room).any():
                    continue
                if len(room_boundary) >= 4:
                    t2 += 1
                    room_boundary = boundary_rotation(room_boundary, theta)
                    room_boundary = normalize(room_boundary, boundary_non_norm, 'boundary')
                    param.append(
                        {'param': room_boundary, 'uid': f'{index:06}_{t2}'}
                    )
    return param

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

def gen_data(ret, if_rotation=True):
    boundary_data = []
    profile_data = []
    area_ratios = []
    # theta_list = [0, np.pi, np.pi / 2, -np.pi / 2] if if_rotation else [0]
    theta_list = [0] if if_rotation else [0]

    # 使用 tqdm 来显示进度条
    for i, key in enumerate(tqdm(list(ret.keys()), desc='Generating data')):

        break_outer = False
        for theta_id, theta in enumerate(theta_list):
            # 处理boundary数据
            boundary = ret[key]['boundary'][:, :2]
            room_boundaries = ret[key]['room_boundaries']
            for room_boundary in room_boundaries:
                room = np.array(room_boundary, dtype=np.float32)
                if np.isnan(room).any() or len(room_boundary) < 4:
                    break_outer = True
                    break
            if break_outer:
                break

            boxes_aligned = ret[key]['box']
            types = ret[key]['type']
            #room_type = assign_types(room_boundaries, boxes_aligned, types)
            room_type = ret[key]['type']

            boundary_non_norm = boundary_rotation(boundary, boundary, theta)
            boundary_param = normalize(boundary_non_norm, boundary_non_norm, 'boundary')
            #FrontDoor, FrontDoor_bbox = create_rectangle_from_two_points(boundary_param[0], boundary_param[1])
            #FrontDoor_type = np.insert(FrontDoor, 0, np.array([[0, 0]]), axis=0)
            #boundary_data.append({'param': FrontDoor_type.astype(np.int32), 'uid': f'{i * 4 + theta_id:06}_0'})

            #是否存在相同的点
            unique_points, counts = np.unique(boundary_param, axis=0, return_counts=True)
            duplicates = unique_points[counts > 1]

            if duplicates.size > 0:
                #print(boundary_param)
                continue

            boundary_param_type = np.insert(boundary_param, 0, np.array([[1, 1]]), axis=0)
            boundary_data.append({'param': boundary_param_type.astype(np.int32), 'uid': f'{i + theta_id:06}_0'})
            duplicates_flag = False
            data_boxes = []

            points_x, points_y = boundary_param.T

            # 分别求 x 和 y 的最小值和最大值
            min_x = np.min(points_x)
            max_x = np.max(points_x)
            min_y = np.min(points_y)
            max_y = np.max(points_y)

            box = np.stack([min_x, min_y, max_x, max_y]).T
            box_type = np.insert(box, 0,  1)
            data_boxes.append(box_type.astype(np.int32))

            temp_room_boundary = []
            temp_profile = []

            for j, room_boundary in enumerate(room_boundaries):
                room = np.array(room_boundary, dtype=np.float32)
                if np.isnan(room).any() or len(room_boundary) < 4:
                    continue
                room_boundary = boundary_rotation(room_boundary, boundary, theta)

                area_raw  = Polygon(room_boundary).area
                room_boundary = normalize(room_boundary, boundary_non_norm, 'boundary')
                area_norm = Polygon(room_boundary).area
                if area_norm > 0:
                    area_ratios.append(area_raw / area_norm)

                unique_points, counts = np.unique(room_boundary, axis=0, return_counts=True)
                duplicates = unique_points[counts > 1]
                if duplicates.size > 0:
                    duplicates_flag = True
                    #print(room_boundary)
                    continue
                points_x, points_y = room_boundary.T
                # 分别求 x 和 y 的最小值和最大值
                min_x = np.min(points_x)
                max_x = np.max(points_x)
                min_y = np.min(points_y)
                max_y = np.max(points_y)

                box = np.stack([min_x,min_y,max_x,max_y]).T
                box_type = np.insert(box, 0, room_type[j] + 1)
                data_boxes.append(box_type.astype(np.int32))
                temp_profile.append(box_type.astype(np.int32))

                room_boundary_type = np.insert(room_boundary, 0, np.array([[room_type[j]+1, room_type[j]+1]]), axis=0)
                temp_room_boundary.append(room_boundary_type.astype(np.int32))

            if duplicates_flag:
                continue

            def extract_sort_key(profile):
                # 提取 profile 中的 min_x 和 min_y
                type, min_x, min_y, max_x, max_y = profile
                return min_x, min_y

            sorted_indices = sorted(range(len(temp_profile)), key=lambda i: extract_sort_key(temp_profile[i]))

            # 根据排序后的索引调整 temp_profile 和 temp_room_boundary 的顺序
            sorted_temp_profile = [temp_profile[i] for i in sorted_indices]
            sorted_temp_room_boundary = [temp_room_boundary[i] for i in sorted_indices]

            sorted_temp_profile.insert(0, data_boxes[0])
            for j in range(len(sorted_temp_room_boundary)):
                boundary_data.append({'param': sorted_temp_room_boundary[j].astype(np.int32), 'uid': f'{i + theta_id:06}_{j+1}'})
            profile_data.append({'profile': sorted_temp_profile, 'uid': f'{i + theta_id:06}'})

            # boundary_data.append(
            #     {'param': room_boundary_type.astype(np.int32), 'uid': f'{i * 4 + theta_id:06}_{index}'})
            # profile_data.append({'profile': data_boxes, 'uid': f'{i * 4 + theta_id:06}'})
    mean_area_ratio = np.mean(area_ratios)
    print("平均面积比例（normalize 前 / 后）:", mean_area_ratio)
    return boundary_data, profile_data

if __name__ == '__main__':

    # loop数据  room_boundaries数据
    # profile数据  data[boxes]数据

    # 数据划分：
    # 	loop下分为train/val/test
    # 	profile下分为train/val/test
    # 	两部分数据分开

    # loop数据：
    # 	[{'param'：[[x0,y0], [x1,y1],....], 'uid':'00001_0' },
    # 	 {'param'：[[x0,y0], [x1,y1],....], 'uid':'00001_1' },
    # 	 .....]

    # profile数据:
    # 	[{'profile':[[x0,y0,x1,y1],[x0,y0,x1,y1]], 'uid':'00001'},
    # 	 {'profile':[[x0,y0,x1,y1],[x0,y0,x1,y1], ....], 'uid':'00002'},
    # 	 ......]

    # profile的uid按照场景id给出，profile数据就是每个房间的包围盒。loop的uid对应profile，每个房间给loop：00001_0,00001_1,这里loop的顺序和profile不需要对应。但每个都要给出唯一的uid。

    #RPLAN数据集
    ret = {}
    for i in range(8):
        with open(f'data{i}.pkl', 'rb') as f:
            temp_dict = pickle.load(f)
            ret.update(temp_dict)

    file_path = 'boundary.pkl'
    # boundary = gen_boundary_data(ret)
    boundary, profile = gen_data(ret)
    # 划分数据集
    total_size = len(boundary)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # 计算训练集、验证集和测试集的索引
    train_indices = range(train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, total_size)

    # 根据索引划分数据集
    train_boundary = [boundary[i] for i in train_indices]
    val_boundary = [boundary[i] for i in val_indices]
    test_boundary = [boundary[i] for i in test_indices]

    # 保存数据集
    file_paths = ['loop\\train.pkl', 'loop\\val.pkl', 'loop\\test.pkl']
    datasets = [train_boundary, val_boundary, test_boundary]

    for path, dataset in zip(file_paths, datasets):
        with open(path, 'wb') as file:
            pickle.dump(dataset, file)

    print("数据集已成功划分并按顺序保存为 train.pkl, val.pkl 和 test.pkl")

    file_path = 'profile.pkl'
    # profile = gen_profile_data(ret)
    total_size = len(profile)
    train_size = int(0.7 * len(profile))
    val_size = int(0.2 * len(profile))
    test_size = len(profile) - train_size - val_size

    # 计算训练集、验证集和测试集的索引
    train_indices = range(train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, total_size)

    # 根据索引划分数据集
    train_boundary = [profile[i] for i in train_indices]
    val_boundary = [profile[i] for i in val_indices]
    test_boundary = [profile[i] for i in test_indices]

    # 保存数据集
    file_paths = ['profile\\train.pkl', 'profile\\val.pkl', 'profile\\test.pkl']
    boundaries = [train_boundary, val_boundary, test_boundary]

    for file_path, boundary in zip(file_paths, boundaries):
        with open(file_path, 'wb') as file:
            pickle.dump(boundary, file)

    print("数据集已成功划分并保存为 train.pkl, val.pkl 和 test.pkl")

    #LIFULL数据集
    # with open('LIFULL.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #boundary, profile = gen_data(ret)

    # # 划分数据集
    # total_size = len(boundary)
    # train_size = int(0.7 * total_size)
    # val_size = int(0.2 * total_size)
    # test_size = total_size - train_size - val_size
    #
    # # 计算训练集、验证集和测试集的索引
    # train_indices = range(train_size)
    # val_indices = range(train_size, train_size + val_size)
    # test_indices = range(train_size + val_size, total_size)
    #
    # # 根据索引划分数据集
    # train_boundary = [boundary[i] for i in train_indices]
    # val_boundary = [boundary[i] for i in val_indices]
    # test_boundary = [boundary[i] for i in test_indices]
    #
    # # 保存数据集
    # file_paths = ['LIFULL_loop\\train.pkl', 'LIFULL_loop\\val.pkl', 'LIFULL_loop\\test.pkl']
    # datasets = [train_boundary, val_boundary, test_boundary]
    #
    # for path, dataset in zip(file_paths, datasets):
    #     with open(path, 'wb') as file:
    #         pickle.dump(dataset, file)
    #
    # print("数据集已成功划分并按顺序保存为 train.pkl, val.pkl 和 test.pkl")
    # #
    # # profile = gen_profile_data(ret)
    # total_size = len(profile)
    # train_size = int(0.7 * len(profile))
    # val_size = int(0.2 * len(profile))
    # test_size = len(profile) - train_size - val_size
    #
    # # 计算训练集、验证集和测试集的索引
    # train_indices = range(train_size)
    # val_indices = range(train_size, train_size + val_size)
    # test_indices = range(train_size + val_size, total_size)
    #
    # # 根据索引划分数据集
    # train_boundary = [profile[i] for i in train_indices]
    # val_boundary = [profile[i] for i in val_indices]
    # test_boundary = [profile[i] for i in test_indices]
    #
    # # 保存数据集
    # file_paths = ['LIFULL_profile\\train.pkl', 'LIFULL_profile\\val.pkl', 'LIFULL_profile\\test.pkl']
    # boundaries = [train_boundary, val_boundary, test_boundary]
    #
    # for file_path, boundary in zip(file_paths, boundaries):
    #     with open(file_path, 'wb') as file:
    #         pickle.dump(boundary, file)
    #
    # print("数据集已成功划分并保存为 train.pkl, val.pkl 和 test.pkl")

