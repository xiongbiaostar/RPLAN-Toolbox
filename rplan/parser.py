import numpy as np
from pyparsing import results


def get_normal_vector(p1, p2):
    """计算从 p1 到 p2 向量的法向量"""
    return np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])

def normalize(vector):
    """标准化向量"""
    norm = np.linalg.norm(vector)
    if norm == 0:
       return vector
    return vector / norm


class FeatureParser(object):

    def __init__(self, type, boxes, room_boundaries, version=1):
        self.type = type
        self.bboxes = boxes
        self.room_boundaries = room_boundaries
        self.version = version

        self.walls = []
        self.index = 0

    def add_wall_if_unique(self, start_point, end_point):
        """添加墙，如果起始点和终点是唯一的"""
        for wall in self.walls:
            if np.array_equal(start_point, wall[0]) and np.array_equal(end_point, wall[1]):
                return False  # 如果已经存在相同的线段，则不添加
            if np.array_equal(start_point, wall[1]) and np.array_equal(end_point, wall[0]):
                return False  # 检查反向的情况
        self.walls.append([start_point, end_point])
        return True


    def _get_walls(self, room_boundaries, thickness=6):

        walls = []

        for wall in self.room_boundaries:
            n = len(wall)
            for i in range(n):
                p1 = wall[i]
                p2 = wall[(i + 1) % n]

                orientation = normalize(get_normal_vector(p1, p2))
                norm_vector = orientation * thickness / 2

                # 计算起始点和终点
                start_point = p1 + norm_vector
                end_point = p1 - norm_vector
                length = max(abs(end_point[0]-start_point[0]), abs(end_point[1]-start_point[1]))

                flag = self.add_wall_if_unique(start_point, end_point)
                if flag:
                    result = {"id" : self.index,
                              "toplevel" : 1,
                              "bottomlevel" : 0,
                              "height" : 0,
                              "length" : length,
                              "thickness" : thickness,
                              "botOffset" : 0,
                              "topOffset" : 0,
                              "side" : 0,
                              "orientation" : { "x" : orientation[0],
                                                "y" : orientation[1],
                                                "z" : 0 },
                              "begin": { "x" : start_point[0],
                                         "y" : start_point[1],
                                         "z" : 0 },
                              "end": { "x" : end_point[0],
                                       "y" : end_point[1],
                                       "z" : 0 } }
                    self.index += 1
                    walls.append({"wall":result})
        return walls

    #def _ger_doors(self):


    #def _get_windows(self):


    def parser(self):

        result = {}


        toplevel = {"level":{"id": 1,
                             "height": 0}}
        bottomlevel = {"level":{"id": 0,
                                "height": 0}}

        result.update({"version": self.version})
        result.update({"toplevel":toplevel})
        result.update({"bottomlevel":bottomlevel})
        walls = self._get_walls(self.room_boundaries)

        results = []
        results.append(result)
        results.append(walls)
        return results





