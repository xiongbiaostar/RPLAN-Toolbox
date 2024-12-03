from dataload import load_data
import pandas as pd
# 描述性统计
if __name__ == '__main__':
    data = load_data('data/')
    boundary_num = []
    box_num = [] 
    box_boundary_num = []
    for _ in data.keys():
        boundary_num.append(data[_]['boundary'].shape[0])
        box_num.append(len(data[_]['room_boundaries']))

        box_boundary_num_temp = 0
        for j in data[_]['room_boundaries']:
            box_boundary_num_temp += j.shape[0]
        
        box_boundary_num.append(box_boundary_num_temp)

    print('外边框点数统计')
    print(pd.Series(boundary_num).value_counts().sort_index())
    print('房间数统计')
    print(pd.Series(box_num).value_counts().sort_index())
    print('房间边框点数统计')
    print(pd.Series(box_boundary_num).value_counts().sort_index())