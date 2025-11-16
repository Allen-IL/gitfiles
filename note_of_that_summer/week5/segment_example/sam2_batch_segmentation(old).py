import os
import cv2
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt

def visualize_mask(image, masks, save_path):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    # 假设 masks 是 tuple，第一个元素是掩码堆叠
    masks_tensor = masks[0]
    for i in range(masks_tensor.shape[0]):
        mask = masks_tensor[i].astype(bool)
        plt.imshow(mask, alpha=0.5)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


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

