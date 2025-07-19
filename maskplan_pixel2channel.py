import os

import cv2
import numpy as np
from skimage import measure,io


CATEGORY_COLOR_MAP = {
    (0, 255, 255): 0,  # 客厅
    (255, 0, 0,): 1,  # 卧室
    (255, 0, 255): 2,  # 厨房
    (0, 0, 255): 3,  # 浴室
    (255, 127, 127): 4,  # 阳台
    (255, 255, 0,): 5,  #
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

# MASKPLAN
# 主要房间类别（排除外墙和前门）
ROOM_COLOR_KEYS = [k for k in Grpah2plan_CATEGORY_COLOR_MAP if Grpah2plan_CATEGORY_COLOR_MAP[k] < 6]

def parse_floorplan_rgb(image_path, gt_path=None):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    boundary = np.zeros((h, w), dtype=np.uint8)
    category = np.full((h, w), 13, dtype=np.uint8)
    instance = np.zeros((h, w), dtype=np.uint8)
    inside = np.zeros((h, w), dtype=np.uint8)

    # 1. 提取 boundary：外墙设为127，前门设为255
    if gt_path:
        gt_image = io.imread(gt_path)  # shape: (256, 256, 4)
        wall_mask_256 = (gt_image[..., 0] == 127).astype(np.uint8)
        door_mask_256 = (gt_image[..., 0] == 255).astype(np.uint8)

        # 缩放到与当前图相同尺寸
        wall_mask_128 = cv2.resize(wall_mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
        door_mask_128 = cv2.resize(door_mask_256, (w, h), interpolation=cv2.INTER_NEAREST)

        kernel = np.ones((2, 2), np.uint8)  # 3x3 的膨胀核
        wall_mask_dilated = cv2.dilate(wall_mask_128, kernel, iterations=1)
        door_mask_dilated = cv2.dilate(door_mask_128, kernel, iterations=1)

        # 将膨胀后的掩码赋值到 boundary 图像
        boundary[wall_mask_dilated == 1] = 127
        boundary[door_mask_dilated == 1] = 255


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
    # return output

    output_resized = np.zeros((128, 128, 4), dtype=output.dtype)
    for i in range(4):
        output_resized[..., i] = cv2.resize(output[..., i], (128, 128), interpolation=cv2.INTER_NEAREST)

    return output_resized

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
    # return output

    output_resized = np.zeros((128, 128, 4), dtype=output.dtype)
    for i in range(4):
        output_resized[..., i] = cv2.resize(output[..., i], (128, 128), interpolation=cv2.INTER_NEAREST)

    return output_resized


if __name__ == "__main__":
    # image_path = "./MASKPLAN/post_19/"
    # save_path = "./MASKPLAN/MASKPLAN_4channel_post19"
    # image_path = "./Graph2plan/data"
    # save_path = "./Graph2plan/G2P_4channel"

    image_path = "./GSDiff/test_gt"
    save_path = "./GSDiff/output_4channel_gt"
    # gt_path = "./dataset/floorplan_dataset"


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for filename in os.listdir(image_path):
        if filename.endswith(".png"):
            # 构造完整的图片路径
            full_image_path = os.path.join(image_path, filename)
            #full_gt_path = os.path.join(gt_path, filename)
            output = parse_floorplan_rgb(full_image_path)

            # 构造保存路径
            save_filename = os.path.join(save_path, filename)
            # 保存处理后的图片
            cv2.imwrite(save_filename, output)
            print(save_filename)

