import torch
from train_ibot import DataAugmentationiBOT, MaskingGenerator
from torchvision.utils import save_image
from PIL import Image
import numpy as np

# 1. 模拟你的数据增强
aug = DataAugmentationiBOT(
    global_crops_scale=(0.14, 1.),
    local_crops_scale=(0.05, 0.4),
    local_crops_number=6,
    image_size=96
)

real_img_path = "./data/part1/train/13.jpg"  
# 如果找不到路径，可以用 glob 自动找一张
import glob
try:
    # real_img_path = glob.glob("./data/part1/*.jpg")[0] # 或者是 .png
    img = Image.open(real_img_path).convert('RGB')
    print(f"成功加载图片: {real_img_path}")
except:
    print("找不到图片，请手动指定 real_img_path")
    # 仅作为最后的 fallback
    img = Image.fromarray(np.uint8(np.random.rand(96, 96, 3) * 255)).convert('RGB')

# 3. 运行增强
crops = aug(img) # 返回 2 global + 6 local

# 4. 模拟 Mask 生成
mask_gen = MaskingGenerator(
    input_size=(12, 12), # 96 // 8 = 12
    num_masking_patches=45
)
mask = mask_gen() # numpy array (12, 12)

# 5. 反归一化以便可视化
def denorm(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

# 6. 保存结果
debug_imgs = []
# Global Crop 1 (Student View)
debug_imgs.append(denorm(crops[0])) 
# Global Crop 2 (Teacher View)
debug_imgs.append(denorm(crops[1]))

# 将 Mask 覆盖在 Global Crop 1 上看看效果
mask_tensor = torch.from_numpy(mask).repeat_interleave(8, dim=0).repeat_interleave(8, dim=1) # upsample to 96x96
masked_img = crops[0].clone()
masked_img[:, mask_tensor==1] = 0 # 把 mask 区域变黑
debug_imgs.append(denorm(masked_img))

save_image(torch.stack(debug_imgs), "debug_view.png")
print("已生成 debug_view.png，请检查：\n1. 图片是否清晰？(如果是一团模糊，说明 Gaussian Blur 半径太大)\n2. Mask 是否覆盖了部分区域？")