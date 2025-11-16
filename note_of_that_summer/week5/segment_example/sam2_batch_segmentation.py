import os
import cv2
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt

def visualize_mask(image, masks, save_path):
    import numpy as np
    import random

    # 1. 原图（RGB → BGR for cv2保存）
    original_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 2. 分割掩码（多实例彩色版）
    masks_tensor = masks[0]
    h, w = image.shape[:2]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # 为每个mask随机分配一种鲜艳颜色
    random.seed(42)  # 固定随机种子，结果可复现
    for i in range(masks_tensor.shape[0]):
        mask = masks_tensor[i].astype(bool)
        color = [random.randint(50, 255) for _ in range(3)]  # 避免太暗
        color_mask[mask] = color

    mask_bgr = color_mask.copy()

    # 3. 叠加效果图（多实例彩色半透明叠加）
    overlay_rgb = image.copy()
    overlay_rgb = (0.5 * overlay_rgb + 0.5 * color_mask).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    # 4. 拼接（竖直方向）
    separator = np.full((5, w, 3), 128, dtype=np.uint8)  # 灰色分隔线
    combined = np.vstack([original_bgr, separator,
                          mask_bgr, separator,
                          overlay_bgr])

    cv2.imwrite(save_path, combined)


def main():
    device = "cpu"  # 使用CPU版本

    checkpoint = "checkpoints/sam2.1_hiera_small.pt"
    config = "configs/sam2.1/sam2.1_hiera_s.yaml"

    sam_model = build_sam2(config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam_model)

    image_dir = "images"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)
        masks = predictor.predict()

        print(f"处理图片 {filename}")
        print(f"  type(masks) = {type(masks)}")
        if isinstance(masks, (list, tuple)):
            print(f"  len(masks) = {len(masks)}")
            if len(masks) > 0:
                print(f"  type(masks[0]) = {type(masks[0])}")
                if hasattr(masks[0], 'shape'):
                    print(f"  masks[0].shape = {masks[0].shape}")
                else:
                    print(f"  masks[0] keys = {list(masks[0].keys()) if isinstance(masks[0], dict) else 'N/A'}")
        else:
            print(f"  masks.shape = {getattr(masks, 'shape', 'N/A')}")

        save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_overlay.png")
        visualize_mask(image, masks, save_path)

if __name__ == "__main__":
    main()

