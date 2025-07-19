import os

from matplotlib import pyplot as plt
from tqdm import tqdm

from rplan.floorplan import Floorplan
from rplan.align import align_fp_gt
from rplan.decorate import get_dw
from rplan.plot import get_figure,get_axes,plot_fp


RPLAN_DIR = './GSDiff/output_4channel'
OUTPUT_DIR = './GSDiff/output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
all_files = [f for f in os.listdir(RPLAN_DIR) if f.endswith(".png")]
for filename in tqdm(all_files, desc="Processing files", unit="file"):
    try:
        file_path = os.path.join(RPLAN_DIR, filename)

        # 处理逻辑
        fp = Floorplan(file_path)
        data = fp.to_dict()

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
        plot_fp(data['boundary'], data['boxes_aligned'][order], data['types'][order], data['doors'], data['windows'],
                ax=ax)

        # 设置绘图范围
        x_vals = data['boundary'][:, 0]
        y_vals = data['boundary'][:, 1]
        xmin, xmax = x_vals.min(), x_vals.max()
        ymin, ymax = y_vals.min(), y_vals.max()
        ax.set_xlim(xmin - 6, xmax + 6)
        ax.set_ylim(ymin - 6, ymax + 6)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

        fig.canvas.draw()
        fig.canvas.print_figure(f"{OUTPUT_DIR}/{os.path.splitext(filename)[0]}.png")
        # 保存绘图结果
        # save_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".png")
        # plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as e:
        print(e)