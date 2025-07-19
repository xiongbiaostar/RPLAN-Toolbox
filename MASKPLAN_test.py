import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from rplan.floorplan import Floorplan
from rplan.align import align_fp_gt
from rplan.decorate import get_dw
from rplan.plot import get_figure, get_axes, plot_fp_rplan, plot_fp_maskplan


RPLAN_DIR = './MASKPLAN/MASKPLAN_4channel'
OUTPUT_DIR = './MASKPLAN/output_3000'
GT_DIR = './dataset/floorplan_dataset'
SAVE_DIR = './MASKPLAN/output_gt_3000'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
all_files = [f for f in os.listdir(RPLAN_DIR) if f.endswith(".png")]

mse_S_all = 0
count = 0
# for filename in tqdm(all_files, desc="Processing files", unit="file"):
#     try:
#
#         floorplan = Image.open(f'{RPLAN_DIR}/{filename}')
#         input_map = np.asarray(floorplan, dtype=np.uint8)
#
#         floorplan = Image.open(f'{GT_DIR}/{filename}')
#         gt_map = np.asarray(floorplan, dtype=np.uint8)
#         gt_map = np.array(Image.fromarray(gt_map).resize((128, 128), Image.NEAREST))
#         cIdx = np.array([1, 2, 3, 4, 1, 2, 2, 2, 2, 5, 1, 6, 1, 10, 7, 8, 9, 10]) - 1
#         gt_cls = cIdx[gt_map[:, :, 1]]
#
#         #计算gt_map[:,:,1]和category_map上对应像素值数量的MSE
#         # 1. 统计 ground-truth 各类别的像素数量
#         gt_counts = {}
#         for val in np.unique(gt_cls):
#             gt_counts[val] = int(np.sum(gt_cls == val))
#
#         # 2. 统计预测结果各类别的像素数量
#         pred_counts = {}
#         for val in np.unique(input_map[:, :, 1]):
#             pred_counts[val] = np.sum(input_map[:, :, 1] == val)
#
#         # 3. 合并所有出现过的类别，保证两个 dict 的 key 一致
#         target_classes = [0, 1, 2, 3, 4, 5]
#
#         gt_counts = {k: v for k, v in gt_counts.items() if k in target_classes}
#         pred_counts = {k: v for k, v in pred_counts.items() if k in target_classes}
#
#         # 构造等长向量（按 0–5 顺序）
#         gt_vec = np.array([gt_counts.get(c, 0) for c in target_classes], dtype=np.float64)
#         pred_vec = np.array([pred_counts.get(c, 0) for c in target_classes], dtype=np.float64)
#
#         # 5. 手工计算 MSE
#         mse = np.mean(((gt_vec - pred_vec) * (20/256) ** 2)** 2)
#         print(f'mse:{mse}')
#         mse_S_all += mse
#         count += 1
#     except Exception as e:
#         print(e)
#
# print(f'mse_S_all:{mse_S_all / count}')

for filename in tqdm(all_files, desc="Processing files", unit="file"):
    try:
        file_path = os.path.join(RPLAN_DIR, filename)

        # 处理逻辑
        # fp = Floorplan(file_path)
        # data = fp.to_dict()
        #
        # boxes_aligned, order, room_boundaries = align_fp_gt(data['boundary'], data['boxes'], data['types'],
        #                                                     data['edges'])
        # data['boxes_aligned'] = boxes_aligned
        # data['order'] = order
        # data['room_boundaries'] = room_boundaries
        #
        # doors, windows = get_dw(data)
        # data['doors'] = doors
        # data['windows'] = windows
        #
        # # 创建绘图
        # fig = get_figure([512, 512])
        # ax = get_axes(fig=fig, rect=[0, 0, 1, 1])  # 获取 ax
        # plot_fp_maskplan(data['boundary'], data['boxes_aligned'][order], data['types'][order], data['doors'], data['windows'], ax=ax)
        #
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
        #
        # fig.canvas.draw()
        # fig.canvas.print_figure(f"{OUTPUT_DIR}/{os.path.splitext(filename)[0]}.png")

        #
        gt_path = os.path.join(GT_DIR, filename)
        gt = Floorplan(gt_path)
        gt_data = gt.to_dict()


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

        x_vals = gt_data['boundary'][:, 0]
        y_vals = gt_data['boundary'][:, 1]
        xmin, xmax = x_vals.min(), x_vals.max()
        ymin, ymax = y_vals.min(), y_vals.max()
        ax.set_xlim(xmin - 6, xmax + 6)
        ax.set_ylim(ymin - 6, ymax + 6)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

        fig.canvas.draw()
        fig.canvas.print_figure(f"{SAVE_DIR}/{os.path.splitext(filename)[0]}.png")
    except Exception as e:
        print(e)
