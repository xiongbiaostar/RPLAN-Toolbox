import json
import os

import matplotlib.pyplot as plt

from rplan.floorplan import Floorplan
from rplan.align import align_fp_gt
from rplan.decorate import get_dw
from rplan.measure import compute_tf
from rplan.parser import FeatureParser
from rplan.plot import get_figure,get_axes,plot_category,plot_boundary,plot_graph,plot_fp,plot_tf

RPLAN_DIR = './data'
file_path = f'{RPLAN_DIR}/0.png'
fp = Floorplan(file_path)
data = fp.to_dict()

boxes_aligned, order, room_boundaries = align_fp_gt(data['boundary'],data['boxes'],data['types'],data['edges'])
data['boxes_aligned'] = boxes_aligned
data['order'] = order
data['room_boundaries'] = room_boundaries

doors,windows = get_dw(data)
data['doors'] = doors
data['windows'] = windows


save_path = os.path.join(RPLAN_DIR, "{}.json".format(data['name']))

parser = FeatureParser(data['types'], data['boxes'], room_boundaries)
result = parser.parser()

with open(save_path, 'w') as fp:
    json.dump(result, fp, indent=1)




# fig = get_figure([512,512])
# plot_boundary(data['boundary'],ax=get_axes(fig=fig,rect=[0,0.5,0.5,0.5]))
# ax = plot_category(fp.category,ax=get_axes(fig=fig,rect=[0.5,0.5,0.5,0.5]))
# plot_graph(data['boundary'],data['boxes'],data['types'],data['edges'],ax=ax)
# plot_fp(data['boundary'], data['boxes_aligned'][order], data['types'][order],ax=get_axes(fig=fig,rect=[0,0,0.5,0.5]))
# plot_fp(data['boundary'], data['boxes_aligned'][order], data['types'][order],data['doors'],data['windows'],ax=get_axes(fig=fig,rect=[0.5,0,0.5,0.5]))
# fig.canvas.draw()
# fig.canvas.print_figure('./output/plot.png')
#
# x,y = compute_tf(data['boundary'])
# plot_tf(x,y)
# plt.savefig('./output/tf.png')
# plt.close()
