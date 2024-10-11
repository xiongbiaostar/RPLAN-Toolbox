import numpy as np

ROOM_CLASS = {1: "living room", 2 : "kitchen", 3 : "bedroom", 4 : "bathroom", 5 : "balcony",  7 : "dining room", 8 : "study",
              10 : "store room",  16 : "undefined"}

#6 : "entrance", 15 : "front door",
def get_normal_vector(p1, p2):
    """计算从 p1 到 p2 向量的法向量"""
    return np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])

def are_points_equal(point1, point2):
    return point1[0] == point2[0] and point1[1] == point2[1] and point1[2] == point2[2]

def get_direction_vector(p1, p2):
    """计算p1到p2向量的方向向量"""
    return np.array([p2[0] - p1[0], p2[1] - p1[1], 0])

def normalize(vector):
    """标准化向量"""
    norm = np.linalg.norm(vector)
    if norm == 0:
       return vector
    return vector / norm

class S3DFeatureParser(object):

    def __init__(self, types, boxes, room_boundaries, outwalls, door_point, window_point, top = 2800, bottom =0):
        self.types = types
        self.bboxes = boxes
        self.room_boundaries = room_boundaries          #（n, m ,2） 房间内墙点
        self.outwalls = outwalls                        #(m, 2)
        self.door_point = door_point                    #(n, 4, 2)
        self.window_point = window_point                #(n, 4, 2)
        self.top = top
        self.bottom = bottom

        self.planeLineIndex = []                #记录plane对应的line索引，创建planeLineMatrix
        self.lineJunctionIndex = []             #记录line对应的junction索引，创建lineJunctionMatrix
        self.lineJunctionPoint = []             #记录line对应的端点坐标，用于寻找plane对应的line
        self.junctions = []                     #保留原始2D点，用来记录已经去过的点，确保唯一
        self.walls = []                         #保留墙的起始点和终点，用来记录已经取过的墙，确保唯一
        self.planes = []                        #保留墙的起始点和终点，用来记录已经去过的墙，确保唯一

    def _add_wall_if_unique(self, start_point, end_point):
        """添加墙，如果起始点和终点是唯一的"""
        for wall in self.walls:
            if np.array_equal(start_point, wall[0]) and np.array_equal(end_point, wall[1]):
                return False  # 如果已经存在相同的线段，则不添加
            if np.array_equal(start_point, wall[1]) and np.array_equal(end_point, wall[0]):
                return False  # 检查反向的情况
        self.walls.append([start_point, end_point])
        return True

    def _add_point_if_unique(self, point):
        """
        判断点是否已经保存过
        """
        for vertex in self.junctions:
            if np.array_equal(point, vertex):
                return False

        self.junctions.append(point)
        return True

    def _add_junction_bottom_top(self,point, index_junctions, junctions):
        """
        对于每个交点创建一个地板上的点和天花板上的点
        """
        flag_junction = self._add_point_if_unique(point)
        if flag_junction:
            junction_bottom = {"coordinate": [point[0], point[1], self.bottom],
                               "ID": index_junctions}
            junctions.append(junction_bottom)
            index_junctions += 1
            junction_top = {"coordinate": [point[0], point[1], self.top],
                            "ID": index_junctions}
            junctions.append(junction_top)
            index_junctions += 1
        return index_junctions, junctions

    def _get_junction(self):
        """
        得到json中junction对应参数，
        coordinates 相交点，即每个角点，
        ID
        """
        junctions = []
        index_junctions = 0
        for wall in self.room_boundaries:
            n = len(wall)
            for i in range(n):
                point = wall[i]
                index_junctions, junctinos = self._add_junction_bottom_top(point, index_junctions, junctions)

        for outwall in self.outwalls:
            index_junctions, junctinos = self._add_junction_bottom_top(outwall, index_junctions, junctions)

        for door in self.door_point:
            n = len(door)
            for i in range(n):
                point = door[i]
                index_junctions, junctions = self._add_junction_bottom_top(point, index_junctions, junctions)

        for window in self.window_point:
            n = len(window)
            for i in range(n):
                point = window[i]
                index_junctions, junctions = self._add_junction_bottom_top(point, index_junctions, junctions)

        return junctions

    def _get_lineJunctionIndex(self, start_point, end_point, junctions):
        """
        根据wall的两个端点找到junction中对应索引
        """
        start_index = None
        end_index = None

        found_start = False
        found_end = False
        for junction in junctions:
            if junction['coordinate'] == start_point:
                start_index = junction['ID']
                found_start = True
            if junction['coordinate'] == end_point:
                end_index = junction['ID']
                found_end = True
            if found_start and found_end:
                break
        self.lineJunctionIndex.append([start_index, end_index])
        self.lineJunctionPoint.append([start_point, end_point])

    def _add_lines_bottom_top(self,start_point, end_point, index_lines, lines, junctions):
        """
        对于每条线，创建一个地板上和天花板上的点，并且保存ID对应的端点坐标。
        """
        direction = normalize(get_direction_vector(start_point, end_point)).tolist()

        flag_line = self._add_wall_if_unique(start_point, end_point)
        if flag_line:
            line_bottom = {"direction": direction,
                           "ID": index_lines,
                           "point": [start_point[0], start_point[1], self.bottom]
                           }

            end_point_bottom = [end_point[0], end_point[1], self.bottom]
            self._get_lineJunctionIndex(line_bottom["point"], end_point_bottom, junctions)

            index_lines += 1
            lines.append(line_bottom)
            line_top = {"direction": direction,
                        "ID": index_lines,
                        "point": [start_point[0], start_point[1], self.top]
                        }

            end_point_top = [end_point[0], end_point[1], self.top]
            self._get_lineJunctionIndex(line_top["point"], end_point_top, junctions)

            index_lines += 1
            lines.append(line_top)
        return index_lines, lines

    def _get_lines(self, junctions):
        """
        得到json中line对应参数，
        direction line沿墙起点的方向
        ID
        point     direction起始点

        每个line对应一个地板，一个天花板的点。同时还有一个地板上点垂直向上方向的line
        _get_lines需要先_get_junction得到对应的junctions集合，来找到对应的垂直的墙。
        """
        lines = []
        index_lines = 0
        outwalls = self.outwalls
        n_outwalls = len(outwalls)
        for wall in self.room_boundaries:
            n = len(wall)
            for i in range(n):
                start_point = wall[i]
                end_point = wall[(i + 1) % n]
                index_line, lines = self._add_lines_bottom_top(start_point, end_point, index_lines, lines, junctions)

        for i in range(n_outwalls):
            start_point = outwalls[i]
            end_point = outwalls[(i + 1) % n_outwalls]
            index_line, lines = self._add_lines_bottom_top(start_point, end_point, index_lines, lines, junctions)

        for door in self.door_point:
            n = len(door)
            for i in range(n):
                start_point = door[i]
                end_point = door[(i + 1) % n]
                index_line, lines = self._add_lines_bottom_top(start_point, end_point, index_lines, lines, junctions)

        for window in self.window_point:
            n = len(window)
            for i in range(n):
                start_point = window[i]
                end_point = window[(i + 1) % n]
                index_line, lines = self._add_lines_bottom_top(start_point, end_point, index_lines, lines, junctions)

        for junction in self.junctions:

            line = {"direction": [0, 0, 1],
                    "ID": index_lines,
                    "point": [junction[0], junction[1], self.bottom]
                    }

            start_point = [junction[0], junction[1], self.bottom]
            end_point = [junction[0], junction[1], self.top]
            self._get_lineJunctionIndex(start_point, end_point, junctions)

            index_lines += 1
            lines.append(line)

        return lines

    def _add_plane_if_unique(self,start_point, end_point, index_planes):
        """添加墙，如果起始点和终点是唯一的"""
        for wall in self.planes:
            if np.array_equal(start_point, wall[0]) and np.array_equal(end_point, wall[1]):
                return wall[2]  # 如果已经存在相同的线段，则不添加
            if np.array_equal(start_point, wall[1]) and np.array_equal(end_point, wall[0]):
                return wall[2]  # 检查反向的情况
        self.planes.append([start_point, end_point, index_planes])
        return None

    def _get_semanticsIndex(self):
        """
        第i个房间对应的房间类型
        """
        semanticsIndex = []
        types = self.types
        for i in range(len(types)):
            semanticsIndex.append([i, types[i]])
        return semanticsIndex

    def _add_planes_wall(self, start_point, end_point, index_planes, index, planes, point_wall):
        """
        planes里面type为wall的墙，并且记录对应的lines索引
        """
        index_exist_plane = self._add_plane_if_unique(start_point, end_point, index_planes)

        if index_exist_plane:
            index.append(index_exist_plane)
        else:
            index.append(index_planes)

        if index_exist_plane is None:
            normal = normalize(get_normal_vector(start_point, end_point))
            l1 = np.linalg.norm(normal)
            if l1 == 0:
                l1 = 1e-6
            offset = abs(-np.dot(normal, np.array(start_point))) / l1

            wall = {"offset": offset,
                    "type": "wall",
                    "ID": index_planes,
                    "normal": [normal[0], normal[1], 0]}

            index_wall = self._find_line_indices(self.lineJunctionPoint, point_wall)
            self.planeLineIndex.append(index_wall)

            index_planes += 1
            planes.append(wall)
        return index_planes, planes, index

    def _add_planes_floor_ceilling(self, wall, index_planes, index, planes):
        """
        添加planes里面type为floor和ceiling信息，并记录对应lines的索引
        """
        # floor
        points_bottom = []
        for point in wall:
            points_bottom.append([point[0], point[1], self.bottom])
        index_bottom = self._find_line_indices(self.lineJunctionPoint, points_bottom)
        self.planeLineIndex.append(index_bottom)

        floor = {"offset": self.bottom,
                 "type": "floor",
                 "ID": index_planes,
                 "normal": [0, 0, 1]}

        # if index == index_living_room:
        #     self.living_room_planeID.append(index_planes)
        index.append(index_planes)

        index_planes += 1
        planes.append(floor)

        # ceilling
        points_top = []
        for point in wall:
            points_top.append([point[0], point[1], self.top])
        index_top = self._find_line_indices(self.lineJunctionPoint, points_top)
        self.planeLineIndex.append(index_top)

        ceiling = {"offset": self.top,
                   "type": "ceiling",
                   "ID": index_planes,
                   "normal": [0, 0, -1]}

        # if index == index_living_room:
        #     self.living_room_planeID.append(index_planes)
        index.append(index_planes)

        index_planes += 1
        planes.append(ceiling)

        return index_planes, planes, index, points_bottom, points_top

    def _get_planes(self):
        """
        得到json中planes对应参数
        offset 面沿着法线方向到原点的距离， 如果法线为[0,1,0],那么offset = +/- y的值
        type ： floor wall ceilling
        ID
        normal 面的法线
        """

        planes = []
        index_planes = 0

        outwalls = self.outwalls
        n_outwalls = len(outwalls)
        index_outwalls = []

        door_point = self.door_point
        index_door = []

        window_point = self.window_point
        index_window = []

        index_semantics = self._get_semanticsIndex()


        for index, wall in enumerate(self.room_boundaries):

            index_planes, planes, index_semantics[index], points_bottom, points_top = self._add_planes_floor_ceilling(wall, index_planes, index_semantics[index], planes)

            n = len(wall)

            for i in range(n):
                point_wall = [points_bottom[i], points_top[i], points_top[(i+1) % n], points_bottom[(i+1) % n]]

                start_point = [point_wall[0][0], point_wall[0][1]]
                end_point = [point_wall[-1][0], point_wall[-1][1]]

                index_planes, planes, index_semantics[index] = self._add_planes_wall(start_point, end_point, index_planes, index_semantics[index], planes, point_wall)

        for i in range(n_outwalls):

            start_point = outwalls[i]
            end_point = outwalls[(i+1) % n_outwalls]

            point_wall = [[outwalls[i][0], outwalls[i][1], self.bottom],
                          [outwalls[i][0], outwalls[i][1], self.top],
                          [outwalls[(i+1) % n_outwalls][0], outwalls[(i+1) % n_outwalls][1], self.top],
                          [outwalls[(i+1) % n_outwalls][0], outwalls[(i+1) % n_outwalls][1], self.bottom]]
            index_planes, planes, index_outwalls = self._add_planes_wall(start_point, end_point, index_planes,
                                                                                 index_outwalls, planes,
                                                                                 point_wall)

        for door in door_point:
            index_door_one = []
            index_planes, planes, index_door_one, _, _ = self._add_planes_floor_ceilling(
                door, index_planes, index_door_one, planes)


            for i in range(2):
                start_point = door[2 * i]
                end_point = door[2 * i + 1]

                point_wall = [[door[2 * i][0], door[2 * i][1], self.bottom],
                              [door[2 * i][0], door[2 * i][1], self.top],
                              [door[2 * i + 1][0], door[2 * i + 1][1], self.top],
                              [door[2 * i + 1][0], door[2 * i + 1][1], self.bottom]]
                index_planes, planes, index_door_one = self._add_planes_wall(start_point, end_point, index_planes,
                                                                             index_door_one, planes,
                                                                             point_wall)
            index_door.append(index_door_one)

        for window in window_point:

            index_window_one = []

            index_planes, planes, index_window_one, _, _ = self._add_planes_floor_ceilling(
                window, index_planes, index_window_one, planes)

            for i in range(2):
                start_point = window[2 * i]
                end_point = window[2 * i + 1]

                point_wall = [[window[2 * i][0], window[2 * i][1], self.bottom],
                              [window[2 * i][0], window[2 * i][1], self.top],
                              [window[2 * i + 1][0], window[2 * i + 1][1], self.top],
                              [window[2 * i + 1][0], window[2 * i + 1][1], self.bottom]]

                index_planes, planes, index_window_one = self._add_planes_wall(start_point, end_point, index_planes,
                                                                             index_window_one, planes,
                                                                             point_wall)
            index_window.append(index_window_one)

        return planes, index_semantics, index_outwalls, index_door, index_window

    def _find_line_indices(self, line_junction_point, points_bottom):
        """
        寻找lines对应的索引
        """
        # 初始化索引列表
        indices = []

        # 遍历points_bottom列表，将每两个相邻的点构成一条线段
        for i in range(len(points_bottom)):
            # 获取当前线段的两个端点
            point1 = points_bottom[i]
            point2 = points_bottom[(i + 1) % len(points_bottom)]

            # 检查当前线段是否与linejunctionpoint中的端点匹配
            for index, (start, end) in enumerate(line_junction_point):
                # 检查两种方向的匹配
                if (are_points_equal(start, point1) and are_points_equal(end, point2)) or (are_points_equal(start, point2) and are_points_equal(end, point1)):
                    # 如果匹配，添加索引到列表（避免重复）
                    if index not in indices:
                        indices.append(index)
                    break

        return indices

    def _get_lineJunctionMatrix(self, lines, Junction):
        """
        根据取得的lineJunctionIndex将lineJunctionMatrix对应值赋值为1
        """
        #lineJunctionMatrix = np.zeros((len(lines), len(Junction))).tolist()
        lineJunctionMatrix = [[0 for _ in range(len(Junction))] for _ in range(len(lines))]
        lineJunctionIndex = self.lineJunctionIndex
        for i in range(len(lines)):
            index = lineJunctionIndex[i]
            lineJunctionMatrix[i][index[0]] = 1
            lineJunctionMatrix[i][index[1]] = 1

        return lineJunctionMatrix

    def _get_planeLineMatrix(self, planes, lines):
        #planeLineMatrix = np.zeros((len(planes), len(lines))).tolist()
        planeLineMatrix = [[0 for _ in range(len(lines))] for _ in range(len(planes))]
        planeLineIndex = self.planeLineIndex
        for i in range(len(planes)):
            index = planeLineIndex[i]
            for j in range(len(index)):
                planeLineMatrix[i][index[j]] = 1
        return planeLineMatrix

    def _get_semantics(self, index_semantics, index_outwalls, index_door, index_window):
        """
        保存semantics信息，包括房间类型和door，windows，outwall
        """
        semantics = []
        index = len(index_semantics)
        outwall = {"planeID": index_outwalls,
                   "type": "outwall",
                   "ID": index}
        index += 1
        semantics.append(outwall)

        for semantic in index_semantics:
            type = ROOM_CLASS.get(semantic[1])
            if type == None:
                type = "undefined"
            semantic_plane = {"planeID": semantic[2:],
                              "type": type,
                              "ID": semantic[0]}
            semantics.append(semantic_plane)

        for door in index_door:
            semantic_door = {"planeID": door,
                             "type": "door",
                             "ID": index}
            index += 1
            semantics.append(semantic_door)

        for i,window in enumerate(index_window):
            #infront_door数据也包含在windows中处理（一边连接外墙一边连接房间）
            if i == len(index_window) - 1:
                semantic_window = {"planeID": window,
                                   "type": "door",
                                   "ID": index}
            else:
                semantic_window = {"planeID": window,
                                    "type": "window",
                                    "ID": index}
            index += 1
            semantics.append(semantic_window)

        return semantics

    def parser(self):

        #results = {"lines":[], "semantics":[], "junctions":[], "planes":[], "lineJunctionMatrix":[], "planeLineMatrix":[]}
        results = {}

        junctions = self._get_junction()
        lines = self._get_lines(junctions)
        planes, index_semantics, index_outwalls, index_door, index_window = self._get_planes()
        lineJunctionMatrix = self._get_lineJunctionMatrix(lines, junctions)
        planeLineMatrix = self._get_planeLineMatrix(planes, lines)

        semantics = self._get_semantics(index_semantics, index_outwalls, index_door, index_window)

        results.update({"lines": lines})
        results.update({"semantics": semantics})
        results.update({"junctions": junctions})
        results.update({"planes": planes})
        results.update({"lineJunctionMatrix": lineJunctionMatrix})
        results.update({"planeLineMatrix": planeLineMatrix})

        return results
