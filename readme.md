# RPLAN-json

## 1.annotation.json表示

```
"line":[{"direction":{}, "ID": , "point": []}]			direction表示点延申的方向， 


"semantics":[{"planeID":[], "type": "outwall"  , "ID": } ,

			 {"planeID":[], "type": "window"  , "ID": } ]


"junctions":[{"coordinate": [ , , ,], "ID": } ]     #junction里保存所有点的信息。

"plane": [{"offset": , "type": , "ID": , "normal": ,} ]      type:floor,wall,ceiling, 			

"planeLineMatrix":

"lineJunctionMatrix"     
```

由点Junction，线line，面plane组成，每个点记录交点也就是所有的角点，线只记录一个起始点和方向direction，由planeLineMatrix来记录line对应的两个端点Junction，面只有偏移量offset，type：（floor，wall，ceiling）这三个类型，normal法向量。offset是由法向量到原点的距离决定。



## 2.json解析器

**S3Dparse.py**中主要实现从外墙和每个房间的角点，房间类型，以及窗户和门数据中提取出对应json文件的数据，返回一个result字典。

**S3Dtest.py**主要是对RPLAN的原始点数据进行处理，从中心墙线获取房间角点，外墙角点。门窗数据





### **使用：**

​	将**input_dir**替换为自己的rplan数据集路径

​	将output_dir替换为输出路径

​	index_forma设置文件命名格式，05d是s3d的命名格式

​	maxfiles控制处理数量
