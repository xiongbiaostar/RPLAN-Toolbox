import itertools

class Wall:
    _id_generator = itertools.count(start=1)

    def __init__(self, boundary):
        self.ID = next(self._id_generator)
        self.Vertices = boundary
        self.thickness = 6


    def to_dict(self):
        return {
            'ID': self.ID,
            'Vertices': self.Vertices,
            'Thickness': self.thickness,
        }

class Door:
    _id_generator = itertools.count(start=1)

    def __init__(self,door):
        self.ID = next(self._id_generator)
        self.RoomID = door[0]
        self.PointLeftUp=[door[1],door[2]]
        self.PointRightDown=[door[1]+door[3],door[2]+door[4]]
        self.thickness= 2
        if(door[3]<=door[4]):
            self.direction = "horizontal"
            if (door[4] > 0):
                self.extention = "outward"
            else:
                self.extention = "inward"
        else:
            self.direction = "vertical"
            if (door[3] > 0):
                self.extention = "outward"
            else:
                self.extention = "inward"



    def to_dict(self):
        return {
            'ID': self.ID,
            'RoomID': self.RoomID,
            'PointLeftUp': self.PointLeftUp,
            'PointRightDown': self.PointRightDown,
            "ExtentionDirection":self.extention,
            'Direction': self.direction,
        }

class Window:
    _id_generator = itertools.count(start=1)

    def __init__(self, windows):
        self.ID = next(self._id_generator)
        self.thickness=2
        self.RoomID = windows[0]
        self.PointLeftUp=[windows[1],windows[2]]
        self.PointRightDown=[windows[1]+windows[3],windows[2]+windows[4]]
        if(windows[3]<=windows[4]):
            self.direction = "horizontal"
            if (windows[4] > 0):
                self.extention = "outward"
            else:
                self.extention = "inward"
        else:
            self.direction = "vertical"
            if (windows[3] > 0):
                self.extention = "outward"
            else:
                self.extention = "inward"



    def to_dict(self):
        return {
            'ID': self.ID,
            'thickness':self.thickness,
            'RoomID': self.RoomID,
            'PointLeftUp': self.PointLeftUp,
            'PointRightDown': self.PointRightDown,
            "ExtentionDirection":self.extention,
            'Direction': self.direction,
        }

class Room:
    _id_generator = itertools.count(start=1)

    def __init__(self, rooms):
        self.ID = next(self._id_generator)
        self.xMin= rooms[0]
        self.yMin = rooms[1]
        self.xMax = rooms[2]
        self.yMax = rooms[3]
        self.type = rooms[4]

    def to_dict(self):
        return {
            'ID': self.ID,
            'PointMin': [self.xMin,self.yMin],
            'PointMax': [self.xMax,self.yMax],
            'type': self.type,
        }

class Edge:
    _id_generator = itertools.count(start=1)

    def __init__(self, edge):
        self.ID = next(self._id_generator)
        self.Room1= edge[0]
        self.Room2 = edge[1]
        if(edge[2]==4):self.relation = "Room1 is in Room2"
        elif(edge[2]==5):self.relation = "Room2 is in Room1"
        else:self.relation = edge[2]

    def to_dict(self):
        return {
            'ID': self.ID,
            'Room1': self.Room1,
            'Room2': self.Room2,
            'Relation': self.relation,
        }

