import os

import cv2
import numpy as np
from skimage import measure,io

# 颜色类别映射
IPLAN_CATEGORY_COLOR_MAP = {
    (244, 242, 229): 0,  # 客厅
    (230, 214, 130): 1,  # 卧室
    (224, 134, 131): 2,  # 厨房
    (144, 188, 219): 3,  # 浴室
    (147, 190, 171): 4,  # 阳台
    (225, 175, 131): 5,  #
    (79, 79, 79): 6,     # 外墙
    (255, 225, 25): 7,   # 前门
}



Grpah2plan_CATEGORY_COLOR_MAP = {
    (244, 242, 229): 0,  # 客厅
    (253,244,171): 1,  # 卧室
    (234,216,214): 2,  # 厨房
    (205,233,252): 3,  # 浴室
    (208,216,135): 4,  # 阳台
    (249,222,189): 5,  #
    (79, 79, 79): 6,     # 外墙
    (255, 225, 25): 7,   # 前门
}

# iplan
ROOM_COLOR_KEYS = [k for k in Grpah2plan_CATEGORY_COLOR_MAP if Grpah2plan_CATEGORY_COLOR_MAP[k] < 6]
def parse_floorplan_rgb(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    boundary = np.zeros((h, w), dtype=np.uint8)
    category = np.full((h, w), 13, dtype=np.uint8)
    instance = np.zeros((h, w), dtype=np.uint8)
    inside = np.zeros((h, w), dtype=np.uint8)

    # 1. 提取 boundary：外墙设为127，前门设为255
    for y in range(h):
        for x in range(w):
            color = tuple(img_rgb[y, x])
            if color == (79, 79, 79):        # 外墙
                boundary[y, x] = 127
            elif color == (255, 225, 25):    # 前门
                boundary[y, x] = 255

    # 2. 提取 inside 区域：所有非 boundary 区域都看作候选房间区域
    outer_wall_mask = (boundary > 0).astype(np.uint8)
    # 2. 查找外轮廓（只保留最外层）
    contours, _ = cv2.findContours(outer_wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 创建空图并填充外轮廓区域内部
    inside = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(inside, contours, -1, color=255, thickness=-1)  # thickness=-1 表示“填充”

    # 3. category + instance 提取（对每个颜色分别处理）
    instance_id = 1
    for color in ROOM_COLOR_KEYS:
        cat_id = Grpah2plan_CATEGORY_COLOR_MAP[color]
        mask_color = np.all(img_rgb == color, axis=-1).astype(np.uint8)

        # 连通区域标记
        labeled = measure.label(mask_color, connectivity=1)
        props = measure.regionprops(labeled)

        for prop in props:
            coords = prop.coords
            rr, cc = coords[:, 0], coords[:, 1]
            category[rr, cc] = cat_id
            instance[rr, cc] = instance_id
            instance_id += 1

    # 4. 将 boundary 区域从 category/instance/inside 中剔除
    category[boundary > 0] = 13
    instance[boundary > 0] = 0
    inside[boundary > 0] = 0

    # 5. 组合结果
    output = np.stack([instance, category, boundary, inside], axis=-1)
    return output

if __name__ == "__main__":
    image_path = "./iplan/pred"
    save_path = "./iplan/iplan_4channel_3000"


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for filename in os.listdir(image_path):
        if filename.endswith(".png"):
            # 构造完整的图片路径
            full_image_path = os.path.join(image_path, filename)
            output = parse_floorplan_rgb(full_image_path)

            # 构造保存路径
            save_filename = os.path.join(save_path, filename)
            # 保存处理后的图片
            cv2.imwrite(save_filename, output)
            print(save_filename)
