# # 生成数据代码, 预计1h运行, 如果data文件夹有data{num}.pkl, 则可以不运行
# import matplotlib.pyplot as plt
# import json
# import numpy as np

# from rplan.floorplan import Floorplan
# from rplan.align import align_fp_gt
# from rplan.decorate import get_dw
# from rplan.measure import compute_tf
# from rplan.plot import get_figure,get_axes,plot_category,plot_boundary,plot_graph,plot_fp,plot_tf
# import os
# import tqdm

# RPLAN_DIR = './data'
# for i in range(8):
#     ret = dict()
#     for _ in tqdm.tqdm(range(i * 10000, i * 10000 + 10000)):
#         file_path = f'{RPLAN_DIR}/{_}.png'
#         fp = Floorplan(file_path)
#         img = fp.image
#         data = fp.to_dict()
#         boxes_aligned, order, room_boundaries ,edges= align_fp_gt(data['boundary'],data['boxes'],data['types'],data['edges'])
#         data['boxes_aligned'] = boxes_aligned
#         data['order'] = order
#         data['room_boundaries'] = room_boundaries


#         doors,windows = get_dw(data)
#         # 'doors': doors.tolist(), 
#         # 'windows': windows.tolist(), 
#         ret[_] = {
#                 'boundary': data['boundary'],
#                 'box': data['boxes_aligned'][order],
#                 'type': data['types'][order],
#                 'edges': data['edges'],
#                 'room_boundaries': room_boundaries
#                 }

#     import pickle

#     # 假设 ret 是您要保存的数据对象
#     with open(f'data{i}.pkl', 'wb') as f:  # 以二进制模式打开文件
#         pickle.dump(ret, f)  # 使用 pickle 保存数据

import pickle
# 加载数据
def load_data(path:str) -> dict:
    ret = {}
    for i in range(8):
        with open(f'{path}/data{i}.pkl', 'rb') as f:  # 以二进制读取模式打开文件
            ret = ret | pickle.load(f)  # 从文件中加载数据
    return ret