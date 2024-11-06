import pickle
import numpy as np
import os
def boundary_rotation(data, theta):
    points_x, points_y = data.T
    x = points_x * np.cos(theta) - points_y * np.sin(theta)
    y = points_x * np.sin(theta) + points_y * np.cos(theta)
    return np.stack([x, y]).T

def normalize(data, boundary, type = 'boundary'):
    max_data = boundary.max()
    min_data = boundary.min()
    if type == 'boundary':
        shape = 2
    else:
        shape = 4
    min_data = np.array([min_data] * shape)
    max_data = np.array([max_data] * shape)
    data = (data - min_data) / (max_data - min_data)
    return data

def gen_boundary_data(ret, if_rotation = True):
    param = []
    if if_rotation:
        theta_list = [0, np.pi, np.pi / 2, -np.pi / 2]
    else:
        theta_list = [0]
    t1 = 0
    for i in list(ret.keys()):
        for theta in theta_list:
            t2 = 0
            boundary = ret[i]['boundary'][:, :2]
            room_boundaries = ret[i]['room_boundaries']
            boundary_non_norm = boundary_rotation(boundary, theta)
            boundary = normalize(boundary_non_norm, boundary_non_norm, 'boundary')
            param.append(
                {'param': boundary, 'uid': f'{t1:06}_{t2}'}
            )

            t2 += 1
            for _ in room_boundaries:
                if len(_) != 0:
                    _ = boundary_rotation(_, theta)
                    _ = normalize(_, boundary_non_norm, 'boundary')
                    param.append(
                        {'param': _, 'uid': f'{t1:06}_{t2}'}
                    )
                    t2 += 1
            t1 += 1
    return param

def profile_rotation(data, theta):
    xmin, ymin, xmax, ymax = data.T

    return np.stack([
        xmin * np.cos(theta) - ymin * np.sin(theta),
        xmin * np.sin(theta) + ymin * np.cos(theta),
        xmax * np.cos(theta) - ymax * np.sin(theta),
        xmax * np.sin(theta) + ymax * np.cos(theta),
    ]).T
    

def gen_profile_data(ret, if_rotation = True):
    profile = []
    if if_rotation:
        theta_list = [0, np.pi, np.pi / 2, -np.pi / 2]
    else:
        theta_list = [0]
    t1 = 0
    for i in list(ret.keys()):
        for theta in theta_list:
            t2 = 0
            boundary = ret[i]['boundary'][:, :2]
            xmin, ymin = boundary.min(axis = 0)
            xmax, ymax = boundary.max(axis = 0)
            boundary_non_norm = profile_rotation(
                                np.array([xmin, ymin, xmax, ymax ])
                            , theta)
            boundary_profile = normalize(boundary_non_norm, boundary_non_norm, 'profile')
            
            profile.append(
                    {'param': boundary_profile, 'uid': f'{t1:06}_{t2}'}
                )
            t2 += 1
            for _ in ret[i]['box']:
                if len(_) != 0:
                    _ = profile_rotation(_, theta)
                    _ = normalize(_, boundary_non_norm, 'profile')
                    profile.append(
                        {'param': _, 'uid': f'{t1:06}_{t2}'}
                    )
                t2 += 1
            t1 += 1
    return profile

import pylab as plt
def boundary_plot(data, uid):
    uid_list = [entry for entry in data if uid in entry['uid']]
    plt.figure(figsize=(8, 8))
    for _ in uid_list:
        
        points = _['param']
        points = np.vstack([points, points[0]])

        plt.plot(points[:, 0], points[:, 1], marker='o', linestyle='-', color='b')
        plt.fill(points[:, 0], points[:, 1], 'b', alpha=0.1)  # Light fill for visibility
        plt.axis('equal')
    plt.show()

def profile_plot(data, uid):
    uid_list = [entry for entry in data if uid in entry['uid']]
    plt.figure(figsize=(8, 8))
    for _ in uid_list:
        points = _['param']
        xmin, ymin, xmax, ymax = points
        square_points = np.array([
            [xmin, ymin],
            [xmin, ymax],
            [xmax, ymax],
            [xmax, ymin],
            [xmin, ymin]
        ])

        plt.plot(square_points[:, 0], square_points[:, 1], marker='o', linestyle='-', color='r')
        plt.fill(square_points[:, 0], square_points[:, 1], alpha=0.2)  # Light fill for visibility

        plt.axis('equal')
    plt.show()


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
	
# 生成准备数据
    ret = {}
    for i in range(8):
        with open(f'data{i}.pkl', 'rb') as f:
            ret = ret | pickle.load(f)

    file_path = 'boundary.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            boundary = pickle.load(file)
    else:
        boundary = gen_boundary_data(ret)
        with open(file_path, 'wb') as file:
            pickle.dump(boundary, file)

    file_path = 'profile.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            profile = pickle.load(file)
    else:
        profile = gen_profile_data(ret)
        with open(file_path, 'wb') as file:
            pickle.dump(profile, file)

# 画图函数

    boundary_plot(boundary, '000001')
    profile_plot(profile, '000001')
    