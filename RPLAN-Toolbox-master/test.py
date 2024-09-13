import matplotlib.pyplot as plt
import json
import numpy as np

from rplan.floorplan import Floorplan
from rplan.align import align_fp_gt
from rplan.decorate import get_dw
from rplan.measure import compute_tf
from rplan.plot import get_figure,get_axes,plot_category,plot_boundary,plot_graph,plot_fp,plot_tf
import  output_class

RPLAN_DIR = './data'
file_path = f'{RPLAN_DIR}/0.png'
fp = Floorplan(file_path)
data = fp.to_dict()

# self.front_door = np.array([min_h,min_w,max_h,max_w],dtype=int)
#exterior_boundary表示多边形边缘的LineString对象（一列表顶点）

#对齐边界框和真实边界框
boxes_aligned, order, room_boundaries ,edges= align_fp_gt(data['boundary'],data['boxes'],data['types'],data['edges'])
data['boxes_aligned'] = boxes_aligned
data['order'] = order
data['room_boundaries'] = room_boundaries

print(room_boundaries,'\n')

doors,windows = get_dw(data)
data['doors'] = doors
data['windows'] = windows

def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(i) for i in obj]
    elif isinstance(obj, np.int32):
        return int(obj)
    else:
        return obj

_Rooms = []
rooms=fp._get_rooms()
for room in rooms:
    _Room = output_class.Room(room).to_dict()
    _Rooms.append(_Room)
_Room_dict_serializable = convert_numpy_to_list(_Rooms)
_Room_json = json.dumps(_Room_dict_serializable,indent=4)
with open('output/room.json', 'w') as f:
    f.write(_Room_json)

_Windows = []
for window in windows:
    _Window = output_class.Window(window).to_dict()
    _Windows.append(_Window)
_Window_dict_serializable = convert_numpy_to_list(_Windows)
_Window_json = json.dumps(_Window_dict_serializable,indent=4)
with open('output/window.json', 'w') as f:
    f.write(_Window_json)

#door type : 0(door) / 1(open wall)
_Doors = []
for door in doors:
    _Door = output_class.Door(door).to_dict()
    _Doors.append(_Door)
_Door_dict_serializable = convert_numpy_to_list(_Doors)
_Door_json = json.dumps(_Door_dict_serializable,indent=4)
with open('output/door.json', 'w') as f:
    f.write(_Door_json)

_Edges = []
for edge in edges:
    _Edge = output_class.Edge(edge).to_dict()
    _Edges.append(_Edge)
_Edge_dict_serializable = convert_numpy_to_list(_Edges)
_Edge_json = json.dumps(_Edge_dict_serializable,indent=4)
with open('output/edge.json', 'w') as f:
    f.write(_Edge_json)

_Walls = []
for wall in room_boundaries:
    _Wall = output_class.Wall(wall).to_dict()
    _Walls.append(_Wall)
_Wall_dict_serializable = convert_numpy_to_list(_Walls)
_Wall_json = json.dumps(_Wall_dict_serializable,indent=4)
with open('output/wall.json', 'w') as f:
    f.write(_Wall_json)


#fig = get_figure([512,512])
fig = plt.figure(figsize=[5.12, 5.12], constrained_layout=True)
plot_boundary(data['boundary'],ax=get_axes(fig=fig,rect=[0,0.5,0.5,0.5]))
ax = plot_category(fp.category,ax=get_axes(fig=fig,rect=[0.5,0.5,0.5,0.5]))
plot_graph(data['boundary'],data['boxes'],data['types'],data['edges'],ax=ax)
plot_fp(data['boundary'], data['boxes_aligned'][order], data['types'][order],ax=get_axes(fig=fig,rect=[0,0,0.5,0.5]))
plot_fp(data['boundary'], data['boxes_aligned'][order], data['types'][order],data['doors'],data['windows'],ax=get_axes(fig=fig,rect=[0.5,0,0.5,0.5]))

fig.canvas.draw()
fig.canvas.print_figure('./output/plot.png')

x,y = compute_tf(data['boundary'])
plot_tf(x,y)
plt.savefig('./output/tf.png')
plt.close()
