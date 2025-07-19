import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from rplan.floorplan import Floorplan
from rplan.align import align_fp_gt
from rplan.decorate import get_dw
from rplan.plot import get_figure, get_axes, plot_fp, plot_fp_rplan
import numpy as np

cIdx  = np.array([1,2,3,4,1,2,2,2,2,5,1,6,1,10,7,8,9,10])-1
def map_types(types, cIdx_map):
    return cIdx_map[types]

def compute_room_areas(boxes):
    wh = boxes[:, [2,3]] - boxes[:, [0,1]]
    return wh[:,0] * wh[:,1]

def compute_class_count_vector(types, num_classes):
    vec = np.zeros(num_classes, dtype=int)
    for t in types:
        if 0 <= t < num_classes:
            vec[t] += 1
    return vec

def compute_adjacent_class_vector(types, edges, num_classes):
    vec = np.zeros(num_classes, dtype=int)
    for i, j in edges[:,:2]:
        ti, tj = types[i], types[j]
        if 0 <= ti < num_classes:
            vec[ti] += 1
        if 0 <= tj < num_classes:
            vec[tj] += 1
    return vec

def compute_area_per_class(types, boxes, num_classes):
    areas = compute_room_areas(boxes)
    vec = np.zeros(num_classes, dtype=float)
    for idx, t in enumerate(types):
        if 0 <= t < num_classes:
            vec[t] += areas[idx]
    return vec

def mean_squared_error_custom(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "Shape mismatch in MSE"
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_custom_area(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "Shape mismatch in MSE"
    return np.mean(((y_true - y_pred) * (20 / 256) ** 2) ** 2)

def compute_floorplan_metrics(pred_data, gt_data, cIdx_map):
    # 先做类型映射
    # pred_types_mapped = map_types(pred_data['types'], cIdx_map)
    # gt_types_mapped = map_types(gt_data['types'], cIdx_map)
    num_classes = cIdx_map.max() + 1
    #
    # # 替换 types
    # pred_data = dict(pred_data)
    # gt_data = dict(gt_data)
    # pred_data['types'] = pred_types_mapped
    # gt_data['types'] = gt_types_mapped

    # 1. 类别计数 MSE
    pred_count = compute_class_count_vector(pred_data['types'], num_classes)
    gt_count = compute_class_count_vector(gt_data['types'], num_classes)
    count_mse = mean_squared_error_custom(gt_count, pred_count)

    # 2. 邻接类别统计 MSE
    pred_adj = compute_adjacent_class_vector(pred_data['types'], pred_data['edges'], num_classes)
    gt_adj = compute_adjacent_class_vector(gt_data['types'], gt_data['edges'], num_classes)
    adj_mse = mean_squared_error_custom(gt_adj, pred_adj)

    # 3. 类别面积 MSE
    pred_area = compute_area_per_class(pred_data['types'], pred_data['boxes'], num_classes)
    gt_area = compute_area_per_class(gt_data['types'], gt_data['boxes'], num_classes)
    area_mse = mean_squared_error_custom_area(gt_area, pred_area)

    return {
        'class_count_mse': count_mse,
        'adjacent_class_mse': adj_mse,
        'class_area_mse': area_mse
    }




RPLAN_DIR = './rplan_data/synth_normalization_3000'
GT_DIR = './dataset/floorplan_dataset/'
OUTPUT_DIR = './rplan_data/output_3000'
SAVE_GT_DIR = './rplan_data/output_gt_3000'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(SAVE_GT_DIR):
    os.makedirs(SAVE_GT_DIR)

mse_T_all = 0
mse_A_all = 0
mse_s_ALL = 0
count = 0

all_files = [f for f in os.listdir(RPLAN_DIR) if f.endswith(".png")]
for filename in tqdm(all_files, desc="Processing files", unit="file"):
    # if count > 3000 : break
    try:

        file_path = os.path.join(RPLAN_DIR, filename)
        gt_path = os.path.join(GT_DIR, filename)
        # 处理逻辑
        fp = Floorplan(file_path)
        gt = Floorplan(gt_path)
        data = fp.to_dict()
        gt_data = gt.to_dict()

        # metrics = compute_floorplan_metrics(data, gt_data, cIdx)
        # mse_T_all += metrics['class_count_mse']
        # mse_A_all += metrics['adjacent_class_mse']
        # mse_s_ALL += metrics['class_area_mse']
        # count += 1
        #
        # print(f'total_count:  {count}')
        #
        # print(f'Class count MSE:       {mse_T_all / count:.4f}')
        # print(f'Adjacency matrix MSE:  {mse_A_all / count:.4f}')
        # print(f'Room area MSE:         {mse_s_ALL / count:.4f}')

        boxes_aligned, order, room_boundaries = align_fp_gt(data['boundary'], data['boxes'], data['types'],
                                                            data['edges'])
        data['boxes_aligned'] = boxes_aligned
        data['order'] = order
        data['room_boundaries'] = room_boundaries

        doors, windows = get_dw(data)
        data['doors'] = doors
        data['windows'] = windows

        # 创建绘图
        fig = get_figure([512, 512])
        ax = get_axes(fig=fig, rect=[0, 0, 1, 1])  # 获取 ax
        plot_fp_rplan(data['boundary'], data['boxes_aligned'][order], data['types'][order], data['doors'], data['windows'],ax=ax)
        fig.canvas.draw()
        fig.canvas.print_figure(f"{OUTPUT_DIR}/{os.path.splitext(filename)[0]}.png")


        boxes_aligned, order, room_boundaries = align_fp_gt(gt_data['boundary'], gt_data['boxes'], gt_data['types'],
                                                            gt_data['edges'])
        gt_data['boxes_aligned'] = boxes_aligned
        gt_data['order'] = order
        gt_data['room_boundaries'] = room_boundaries

        doors, windows = get_dw(gt_data)
        gt_data['doors'] = doors
        gt_data['windows'] = windows

        # 创建绘图
        fig = get_figure([512, 512])
        ax = get_axes(fig=fig, rect=[0, 0, 1, 1])  # 获取 ax
        plot_fp_rplan(gt_data['boundary'], gt_data['boxes_aligned'][order], gt_data['types'][order], gt_data['doors'], gt_data['windows'],ax=ax)


        # # 设置绘图范围
        # x_vals = data['boundary'][:, 0]
        # y_vals = data['boundary'][:, 1]
        # xmin, xmax = x_vals.min(), x_vals.max()
        # ymin, ymax = y_vals.min(), y_vals.max()
        # ax.set_xlim(xmin - 6, xmax + 6)
        # ax.set_ylim(ymin - 6, ymax + 6)
        # ax.set_aspect('equal')
        # ax.invert_yaxis()
        # ax.axis('off')

        fig.canvas.draw()
        fig.canvas.print_figure(f"{SAVE_GT_DIR}/{os.path.splitext(filename)[0]}.png")
        # 保存绘图结果
        # save_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".png")
        # plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as e:
        print(e)
