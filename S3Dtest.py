import json
import math
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from rplan.S3Dparse import S3DFeatureParser
from rplan.floorplan import Floorplan
from rplan.align import align_fp_gt
from rplan.decorate import get_dw

def distance(p1, p2):
    """计算两点之间的距离"""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
def is_point_on_line_segment(p, p1, p2):
    """
    判断点 p 是否在线段 p1p2 上
    :param p: 点的坐标 [x, y]
    :param p1: 线段起点坐标 [x, y]
    :param p2: 线段终点坐标 [x, y]
    :return: 如果点在线段上，返回 True，否则返回 False
    """
    # 1. 检查三点是否共线
    if (p2[0] - p1[0]) * (p[1] - p1[1]) != (p[0] - p1[0]) * (p2[1] - p1[1]):
        return False

    # 2. 检查 p 是否在 p1 和 p2 之间（包括端点）
    if min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1]):
        return True

    return False
def get_doors_room_relation(doors, rooms):
    """
    得到门和房间的连接关系，判断门的中心点是否在线上，保存对应的room_boundaries中的位置索引
    """
    doors_edge = []
    for door in doors:
        p = [door[1] + door[3]/2, door[2] + door[4]/2]
        edge = []
        for i in range(len(rooms)):
            room = rooms[i]
            for j in range(len(room)):
                if is_point_on_line_segment(p, room[j], room[(j + 1) % len(room)]):
                    edge.append(i)
                    edge.append(j)
        doors_edge.append(edge)
    return doors_edge

def get_windows_room_relation(windows, rooms, outwall):
    """
    得到窗户和房间，外墙的连接关系，判断窗户的中心点是否在线上，保存对应的room_boundaries，outwalls中的位置索引
    """
    windows_edge = []
    for window in windows:
        p = [window[1] + window[3]/2, window[2] + window[4]/2]
        edge = []
        for i in range(len(rooms)):
            room = rooms[i]
            for j in range(len(room)):
                if is_point_on_line_segment(p, room[j], room[(j + 1) % len(room)]):
                    edge.append(i)
                    edge.append(j)
                    break
        for k in range(len(outwall)):
            if is_point_on_line_segment(p, outwall[k], outwall[(k + 1) % len(outwall)]):
                edge.append(k)
                break
        windows_edge.append(edge)
    return windows_edge

def plot_door(doors, room_boundaries, inner_wall, thickness=6):
    """
    根据门对应的边的索引来找到inner_wall中的位置，来取到门的厚度
    """
    doors_points = []
    doors_edges = get_doors_room_relation(doors, room_boundaries)
    for index, edge in enumerate(doors_edges):
        door = doors[index]
        start_point = door[1:3]
        end_point = door[1:3] + door[3:5]

        if door[3] < door[4]:
            if door[4] > 0:
                start_point[1] = start_point[1] + thickness / 2
            else:
                start_point[1] = start_point[1] - thickness / 2
            start_inner = inner_wall[edge[0]][edge[1]][0]
            start_outer = inner_wall[edge[2]][edge[3]][0]
            doors_points.append([[start_inner, start_point[1]],
                                  [start_inner, end_point[1]],
                                  [start_outer, end_point[1]],
                                  [start_outer, start_point[1]]])

        else:
            if door[3] > 0:
                start_point[0] = start_point[0] + thickness / 2
            else:
                start_point[0] = start_point[0] - thickness / 2
            start_inner = inner_wall[edge[0]][edge[1]][1]
            start_outer = inner_wall[edge[2]][edge[3]][1]
            doors_points.append([[start_point[0], start_inner],
                                 [end_point[0], start_inner],
                                 [end_point[0], start_outer],
                                 [start_point[0], start_outer]])
    return doors_points

def plot_window(doors, room_boundaries, outwalls, inner_wall, outer_wall, thickness=6):
    """
    根据窗户对应的边的索引来找到outer_wall中的位置，来取到窗的厚度
    """
    doors_points = []
    doors_edges = get_windows_room_relation(doors, room_boundaries, outwalls)
    for index, edge in enumerate(doors_edges):
        door = doors[index]
        start_point = door[1:3]
        end_point = door[1:3] + door[3:5]

        if door[3] < door[4]:
            if door[4] > 0:
                start_point[1] = start_point[1] + thickness / 2
            else:
                start_point[1] = start_point[1] - thickness / 2
            start_inner = inner_wall[edge[0]][edge[1]][0]
            start_outer = outer_wall[edge[2]][0]
            doors_points.append([[start_inner, start_point[1]],
                                  [start_inner, end_point[1]],
                                  [start_outer, end_point[1]],
                                  [start_outer, start_point[1]]])

        else:
            if door[3] > 0:
                start_point[0] = start_point[0] + thickness / 2
            else:
                start_point[0] = start_point[0] - thickness / 2
            start_inner = inner_wall[edge[0]][edge[1]][1]
            start_outer = outer_wall[edge[2]][1]
            doors_points.append([[start_point[0], start_inner],
                                 [end_point[0], start_inner],
                                 [end_point[0], start_outer],
                                 [start_point[0], start_outer]])
    return doors_points

def norm(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # 计算门的长度
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
         length = 1e-6
    # 计算单位法向量
    norm_dx = dy / length
    norm_dy = -dx / length

    return norm_dx, norm_dy

def calculate_inner_contours(room_boundaries, thickness=6, min_gap = 0.5):
    """
    通过中心墙线来根据墙的厚度取到房间的角点，

    """
    inner_contours = [point.copy() for point in room_boundaries]

    for i, boundary in enumerate(room_boundaries):
        n = len(boundary)
        j = 0
        while j < n:
            # 获取中心线的两个相邻点
            p1 = boundary[j]
            p2 = boundary[(j + 1) % n]

            norm_dx, norm_dy = norm(p1, p2)

            # 计算内边框的两个相邻点
            inner_contours[i][j] = [inner_contours[i][j][0] + norm_dx * (thickness / 2),
                                    inner_contours[i][j][1] + norm_dy * (thickness / 2)]

            inner_contours[i][(j+1)% n] = [inner_contours[i][(j+1)% n][0] + norm_dx * (thickness / 2),
                                           inner_contours[i][(j+1)% n][1] + norm_dy * (thickness / 2)]

            distance_p = distance(p1, p2)
            p0 = boundary[j - 1]
            p3 = boundary[(j + 2) % n]
            norm_dx_p01, norm_dy_p01 = norm(p0, p1)
            norm_dx_p23, norm_dy_p23 = norm(p2, p3)

            if distance_p <= thickness and (norm_dx_p01 == -norm_dx_p23 and norm_dy_p01 == -norm_dy_p23):
                adjusted_thickness = distance_p - min_gap

                inner_contours[i][j-1] = [inner_contours[i][j-1][0] + norm_dx_p01 * ((adjusted_thickness - thickness) / 2),
                                    inner_contours[i][j-1][1] + norm_dy_p01 * ((adjusted_thickness - thickness) / 2)]
                inner_contours[i][j] = [inner_contours[i][j][0] + norm_dx_p01 * ((adjusted_thickness - thickness) / 2),
                                        inner_contours[i][j][1] + norm_dy_p01 * ((adjusted_thickness - thickness) / 2)]

                inner_contours[i][(j + 1) % n] = [inner_contours[i][(j + 1) % n][0] + norm_dx_p23 * (adjusted_thickness / 2),
                                        inner_contours[i][(j + 1) % n][1] + norm_dy_p23 * (adjusted_thickness / 2)]
                inner_contours[i][(j + 2) % n] = [inner_contours[i][(j + 2) % n][0] + norm_dx_p23 * (adjusted_thickness / 2),
                                        inner_contours[i][(j + 2) % n][1] + norm_dy_p23 * (adjusted_thickness / 2)]
                j += 2
            else:
                j += 1

    return inner_contours

def calculate_outwall_contours(outwalls, thickness=6, min_gap=0.5):
    """
    通过中心墙线根据墙的厚度来取到外墙角点，
    同时判断两个点之间距离小于墙的厚度并且相对，这是将小区域移除，并设置一个新的角点。
    外墙上的前两个点是infront_door的点，有可能和墙共线需要判断共线来确定处理方式
    """
    outwall_contours = [outwall.copy() for outwall in outwalls]
    tolerance = 1e-6

    def is_collinear(p1, p2, p3):
        """判断三点是否共线，通过向量叉积计算面积"""
        return abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) < tolerance

    n = len(outwalls)
    infront_door_1 = is_collinear(outwalls[-1], outwalls[0], outwalls[1])
    infront_door_2 = is_collinear(outwalls[0], outwalls[1], outwalls[2])

    skip_next = False
    for i in range(n):
        if skip_next:
            skip_next = False
            continue
        p1 = outwalls[i]
        p2 = outwalls[(i + 1) % n]

        norm_dx, norm_dy = norm(p1, p2)
        #对于outwalls上前两个点infront_door的处理进行判定，共线只调整一次，不共线说明是角点，正常处理
        if i == 0:
            if not infront_door_1:
                outwall_contours[i] = [outwall_contours[i][0] + norm_dx * (thickness / 2),
                                           outwall_contours[i][1] + norm_dy * (thickness / 2)]

        elif i == 1:
            if not infront_door_2:
                outwall_contours[i] = [outwall_contours[i][0] + norm_dx * (thickness / 2),
                                        outwall_contours[i][1] + norm_dy * (thickness / 2)]

        else:
            outwall_contours[i] = [outwall_contours[i][0] + norm_dx * (thickness / 2),
                                    outwall_contours[i][1] + norm_dy * (thickness / 2)]

        outwall_contours[(i + 1) % n] = [outwall_contours[(i + 1) % n][0] + norm_dx * (thickness / 2),
                                            outwall_contours[(i + 1) % n][1] + norm_dy * (thickness / 2)]

        distance_p = distance(p1, p2)
        p0 = outwalls[i - 1]
        p3 = outwalls[(i + 2) % n]
        norm_dx_p01, norm_dy_p01 = norm(p0, p1)
        norm_dx_p23, norm_dy_p23 = norm(p2, p3)

        if distance_p <= thickness and (norm_dx_p01 == -norm_dx_p23 and norm_dy_p01 == -norm_dy_p23):
            adjusted_thickness = distance_p - thickness /2 - min_gap

            outwall_contours[(i + 1) % n] = [outwall_contours[(i + 1) % n][0] + norm_dx_p23 * adjusted_thickness,
                                             outwall_contours[(i + 1) % n][1] + norm_dy_p23 * adjusted_thickness]
            outwall_contours[(i + 2) % n] = [outwall_contours[(i + 2) % n][0] + norm_dx_p23 * adjusted_thickness,
                                             outwall_contours[(i + 2) % n][1] + norm_dy_p23 * adjusted_thickness]
            skip_next = True
    return outwall_contours

def get_outwall_point(processed_boundary, thickness=6):
    """
    将外墙中心墙线的点取出，同时判断两个点之间距离小于墙的厚度并且相对，这是将小区域移除，并设置一个新的角点。
    """
    n_boundaries = len(processed_boundary)
    outwalls = []
    i = 0
    while i < n_boundaries:
        p1 = processed_boundary[i]
        p2 = processed_boundary[(i + 1) % n_boundaries]
        p0 = processed_boundary[i - 1]
        p3 = processed_boundary[(i + 2) % n_boundaries]
        distance_p = distance(p1, p2)
        norm_dx_p01, norm_dy_p01 = norm(p0, p1)
        norm_dx_p23, norm_dy_p23 = norm(p2, p3)

        if distance_p <= thickness and (norm_dx_p01 == -norm_dx_p23 and norm_dy_p01 == -norm_dy_p23):

            distance_p1 = distance(p0, p1)
            distance_p2 = distance(p2, p3)
            if distance_p1 < distance_p2:
                outwalls.pop()
                adjusted_location = p0[0] if p1[1] == p0[1] else p0[1]
                if p1[0] == p2[0]:
                    pair = [adjusted_location, p2[1]]
                else:
                    pair = [p2[0], adjusted_location]
            else:
                i += 1
                adjusted_location = p3[0] if p2[1] == p3[1] else p3[1]
                if p2[0] == p1[0]:
                    pair = [adjusted_location, p1[1]]
                else:
                    pair = [p1[0], adjusted_location]

            outwalls.append(pair)
            i += 2

        else:
            pair = [processed_boundary[i][0], processed_boundary[i][1]]
            # 将转换后的对添加到结果列表中
            outwalls.append(pair)
            i += 1
    return outwalls

def get_room_wall_point(room_boundaries, thickness=6):
    """
    将中心墙线的点取出，同时判断两个点之间距离小于墙的厚度并且相对，这是将小区域移除，并设置一个新的角点。
    """
    room_wall = []
    for room in room_boundaries:
        wall = []
        n_room = len(room)
        i = 0
        while i < n_room:
            p1 = [int(room[i][0]), int(room[i][1])]
            p2 = [int(room[(i + 1) % n_room][0]), int(room[(i + 1) % n_room][1])]
            p0 = [int(room[i - 1][0]), int(room[i - 1][1])]
            p3 = [int(room[(i + 2) % n_room][0]), int(room[(i + 2) % n_room][1])]
            distance_p = distance(p1, p2)
            norm_dx_p01, norm_dy_p01 = norm(p0, p1)
            norm_dx_p23, norm_dy_p23 = norm(p2, p3)

            if distance_p <= thickness and (norm_dx_p01 == -norm_dx_p23 and norm_dy_p01 == -norm_dy_p23):
                distance_p1 = distance(p0, p1)
                distance_p2 = distance(p2, p3)
                if distance_p1 < distance_p2:
                    if p0 in wall:
                        wall.remove(p0)
                    if p2 in wall:
                        wall.remove(p2)
                    adjusted_location = p0[0] if p1[1] == p0[1] else p0[1]
                    if p1[0] == p2[0]:
                        point = [adjusted_location, p2[1]]
                    else:
                        point = [p2[0], adjusted_location]
                else:
                    if p2 in wall:
                        wall.remove(p2)
                    if p3 in wall:
                        wall.remove(p3)
                    i += 1
                    adjusted_location = p3[0] if p2[1] == p3[1] else p3[1]
                    if p2[0] == p1[0]:
                        point = [adjusted_location, p1[1]]
                    else:
                        point = [p1[0], adjusted_location]
                wall.append(point)
                i += 2
            else:
                wall.append(p1)
                i += 1

        room_wall.append(wall)
    return room_wall

def convert_real(point, origin, scale = 50):
    """
    根据整体的包围盒取中心点作为原点，对坐标进行处理
    再以1：50的比例尺放大点的坐标
    """
    new_point = [point[0] - origin[0], point[1] - origin[1]]
    real_point = [new_point[0] * scale, new_point[1] * scale]

    return real_point

def process_file(args):
    file_name, input_dir, output_dir, index_format = args
    try:
        file_path = os.path.join(input_dir, file_name)
        fp = Floorplan(file_path)
        data = fp.to_dict()

        boxes_aligned, order, room_boundaries = align_fp_gt(data['boundary'],data['boxes'],data['types'],data['edges'])
        data['boxes_aligned'] = boxes_aligned
        data['order'] = order
        data['room_boundaries'] = room_boundaries

        doors,windows = get_dw(data)
        data['doors'] = doors
        data['windows'] = windows

        outwalls = get_outwall_point(data['boundary'])

        #infront_door
        start = []
        dx, dy = 0, 0
        if outwalls[0][0] == outwalls[1][0]:
            start = (outwalls[0][0], min(outwalls[0][1], outwalls[1][1]))
            end = (outwalls[0][0], max(outwalls[0][1], outwalls[1][1]))
            dx, dy = 0, end[1] - start[1]
        elif outwalls[0][1] == outwalls[1][1]:
            start = (min(outwalls[0][0], outwalls[1][0]), outwalls[0][1])
            end = (max(outwalls[0][0], outwalls[1][0]), outwalls[0][1])
            dx, dy = end[0] - start[0], 0

        new_windows = np.vstack((windows, [windows[len(windows)-1][0] + 1, start[0], start[1], dx, dy, 0]))

        room_wall = get_room_wall_point(room_boundaries)

        inner_wall = calculate_inner_contours(room_wall)
        outer_boundaries = calculate_outwall_contours(outwalls)

        update_doors = plot_door(doors, room_wall, inner_wall)
        update_windows = plot_window(new_windows, room_wall, outwalls, inner_wall, outer_boundaries)


        bboxes = np.array(data['boxes'])
        min_x0 = np.min(bboxes[:, 0])
        min_y0 = np.min(bboxes[:, 1])

        max_x1 = np.max(bboxes[:, 2])
        max_y1 = np.max(bboxes[:, 3])

        all_bbox = [min_x0, min_y0, max_x1, max_y1]
        origin = [(min_x0 + max_x1) / 2, (min_y0 + max_y1) / 2]


        #inner_Wall
        real_inner_wall = []
        for wall in inner_wall:
            real_wall = []
            for i in range(len(wall)):
                real_wall.append(convert_real(wall[i], origin))
            real_inner_wall.append(real_wall)

        #outer_boundaries
        real_outer_boundaries = []
        for i in range(len(outer_boundaries)):
            real_outer_boundaries.append(convert_real(outer_boundaries[i], origin))

        #update_doors
        real_update_doors = []
        for door in update_doors:
            real_door = []
            for i in range(len(door)):
                 real_door.append(convert_real(door[i], origin))
            real_update_doors.append(real_door)

        #update_windows
        real_update_windows = []
        for windows in update_windows:
            real_windows = []
            for i in range(len(windows)):
                real_windows.append(convert_real(windows[i], origin))
            real_update_windows.append(real_windows)

        base_name = os.path.splitext(file_name)[0]
        scene_dir = os.path.join(output_dir, f'scene_{index_format.format(int(base_name))}')
        os.makedirs(scene_dir, exist_ok=True)
        output_file_path = os.path.join(scene_dir, 'annotation_3d.json')
        parser = S3DFeatureParser(data['types'], data['boxes'], real_inner_wall, real_outer_boundaries, real_update_doors, real_update_windows)
        result = parser.parser()

        with open(output_file_path, 'w') as fp:
            json.dump(result, fp)
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")


def main(input_dir, output_dir, index_format, maxfiles=-1):
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)
    files = files[:maxfiles]
    args_list = [(file_name, input_dir, output_dir, index_format) for file_name in files]

    with Pool(processes=4) as pool:
        list(tqdm(pool.imap(process_file, args_list), total=len(files), desc="Processing files"))


if __name__ == "__main__":
    input_dir = 'dataset/floorplan_dataset'
    output_dir = 'dataset/structured3d'
    index_format = "{:05d}"
    maxfiles = -1
    main(input_dir, output_dir, index_format, maxfiles)

