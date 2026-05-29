# RPLAN_Toolbox

new_jason分支主要是对于RPLAN数据集进行处理，转化为TLC-PLAN的训练数据，同时包含论文中所有方法的对比可视化的处理代码。

## hnc_data.py

这部分代码主要是对RPLAN数据集以及LIFULL数据集进行处理，将其中的json格式表示提取出来，同时对每个房间进行排序，在保存为pkl数据。用于TLC-PLAN部分的

其中profile_data中保存的是各个房间的bbox数据，存储格式是``[type, min_x,min_y,max_x,max_y]``.

boundary_data中保存的是各个房间的角点数据，存储格式是``[[type, point1, point2, ....], [type, point1, point2, ....]]``.

代码中以7：2：1划分为训练集，测试集和验证集。



## iplan_pixel2channel.py      maskplan_pixel2channel.py

这部分代码是将iplan和maskplan结果中的栅格图转化为RPLAN原始格式的四通道图，以便后续可视化处理过程中加入门窗。



## GSDiff_test.py      Graph2plan_test.py   MASKPLAN_test.py   TLC_test.py   rplan_test.py

这部分代码是对于处理之后的数据转化为论文中的可视化结果代码。
