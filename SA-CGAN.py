import os
import xml.etree.ElementTree as ET
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
from torchvision.ops import deform_conv2d

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========================
# 超参数
# ========================
latent_dim = 256
img_size = (256, 256)
batch_size = 4
num_epochs = 160
grad_accum_steps = 4
use_amp = True
warmup_epochs = 10
adv_ramp_epochs = 20

lr_g = 2e-4
lr_d = 1e-4
betas = (0.0, 0.9)

n_classes = 4
class_names = ["DT_CRACK", "DT_GAP", "DT_LACUNAS", "DT_SUBSIDENCE"]
cond_dim = n_classes + 4

lambda_fm = 2.0
lambda_perceptual = 0.5
lambda_continuity = 1.0
lambda_bbox = 2.0
lambda_focus = 1.5
lambda_edge = 1.2
lambda_bg = 0.7
lambda_detail = 0.9

# bbox-supervised module losses
lambda_att = 1.2
lambda_delta = 1.0
lambda_delta_map = 1.0
lambda_feat_map = 0.8
lambda_gain_floor = 0.02
lambda_crack_boost = 2.5

perceptual_imgsz = 128
preview_count = 4
best_of_k = 2

r1_gamma = 10.0
r1_every = 16

max_grad_norm_d = 1.0
max_grad_norm_g = 1.0

# 断点保存路径
CHECKPOINT_PATH = "checkpoint_enhanced3.pth"
RESULTS_DIR = "generated_results_enhanced3"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========================
# 数据集（不变）
# ========================
class BScanDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = []
        for class_idx, class_name in enumerate(class_names):
            img_dir = os.path.join(root_dir, class_name, "JPEGImages")
            ann_dir = os.path.join(root_dir, class_name, "Annotations")
            if not os.path.exists(img_dir):
                continue
            for img_file in os.listdir(img_dir):
                if img_file.endswith((".jpg", ".png")):
                    img_path = os.path.join(img_dir, img_file)
                    ann_path = os.path.join(ann_dir, os.path.splitext(img_file)[0] + ".xml")
                    if os.path.exists(ann_path):
                        self.data_list.append((img_path, ann_path, class_idx))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, ann_path, label = self.data_list[idx]
        image = Image.open(img_path).convert("L")
        orig_w, orig_h = image.size
        tree = ET.parse(ann_path)
        root = tree.getroot()
        obj = root.find("object")
        if obj is not None:
            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))
        else:
            xmin, ymin, xmax, ymax = 0, 0, orig_w, orig_h
        if self.transform:
            image = self.transform(image)
        condition = torch.zeros(cond_dim, dtype=torch.float32)
        condition[label] = 1.0
        ow = float(max(1, orig_w))
        oh = float(max(1, orig_h))
        x = float(xmin) / ow
        y = float(ymin) / oh
        bw = float(max(0, xmax - xmin)) / ow
        bh = float(max(0, ymax - ymin)) / oh
        condition[n_classes:] = torch.tensor([x, y, bw, bh], dtype=torch.float32)
        return image, condition

def build_class_balanced_preview(dataset):
    selected_indices = [None] * n_classes
    for idx, (_, _, class_idx) in enumerate(dataset.data_list):
        if selected_indices[class_idx] is None:
            selected_indices[class_idx] = idx
        if all(v is not None for v in selected_indices):
            break
    if any(v is None for v in selected_indices):
        missing = [class_names[i] for i, v in enumerate(selected_indices) if v is None]
        raise RuntimeError(f"Preview build failed. Missing classes in dataset: {missing}")
    fixed_real_list = []
    fixed_cond_list = []
    for idx in selected_indices:
        img, cond = dataset[idx]
        fixed_real_list.append(img)
        fixed_cond_list.append(cond)
    fixed_real = torch.stack(fixed_real_list, dim=0)
    fixed_conds = torch.stack(fixed_cond_list, dim=0)
    return fixed_real, fixed_conds

# ========================
# 条件掩码工具（不变）
# ========================
def build_bbox_mask(cond, h, w):
    bbox = cond[:, n_classes:]
    x = bbox[:, 0].clamp(0.0, 1.0)
    y = bbox[:, 1].clamp(0.0, 1.0)
    bw = bbox[:, 2].clamp(0.0, 1.0)
    bh = bbox[:, 3].clamp(0.0, 1.0)
    x1 = (x * w).floor().long()
    y1 = (y * h).floor().long()
    x2 = ((x + bw) * w).ceil().long()
    y2 = ((y + bh) * h).ceil().long()
    x1 = x1.clamp(0, w)
    y1 = y1.clamp(0, h)
    x2 = x2.clamp(0, w)
    y2 = y2.clamp(0, h)
    xs = torch.arange(w, device=cond.device).view(1, 1, 1, w)
    ys = torch.arange(h, device=cond.device).view(1, 1, h, 1)
    x1v = x1.view(-1, 1, 1, 1)
    x2v = x2.view(-1, 1, 1, 1)
    y1v = y1.view(-1, 1, 1, 1)
    y2v = y2.view(-1, 1, 1, 1)
    mask_x = (xs >= x1v) & (xs < x2v)
    mask_y = (ys >= y1v) & (ys < y2v)
    mask = (mask_x & mask_y).to(dtype=cond.dtype)
    return mask

def build_soft_bbox_mask(cond, h, w, blur_ks=9):
    mask = build_bbox_mask(cond, h, w)
    if blur_ks > 1:
        pad = blur_ks // 2
        mask = F.avg_pool2d(mask, kernel_size=blur_ks, stride=1, padding=pad)
    return mask.clamp(0.0, 1.0)

# ========================
# 边缘聚焦目标构建（不变）
# ========================
def build_edge_focused_target(real_img, cond, h, w, blur_ks=3, edge_blur=5):
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=real_img.device, dtype=real_img.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                      device=real_img.device, dtype=real_img.dtype).view(1,1,3,3)
    gx = F.conv2d(real_img, kx, padding=1)
    gy = F.conv2d(real_img, ky, padding=1)
    edge = torch.sqrt(gx*gx + gy*gy + 1e-6)
    if edge.shape[-2] != h or edge.shape[-1] != w:
        edge = F.interpolate(edge, size=(h, w), mode='bilinear', align_corners=False)
    bbox_mask = build_soft_bbox_mask(cond, h=h, w=w, blur_ks=blur_ks)
    edge_in_bbox = edge * bbox_mask
    flat = edge_in_bbox.view(edge_in_bbox.size(0), -1)
    min_v = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    max_v = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    denom = max_v - min_v + 1e-6
    edge_norm = (edge_in_bbox - min_v) / denom
    if edge_blur > 1:
        edge_norm = F.avg_pool2d(edge_norm, kernel_size=edge_blur, stride=1, padding=edge_blur//2)
    return edge_norm.clamp(0.0, 1.0)

# ========================
# 损失函数（不变，略）
# ========================
class PerceptualLoss(nn.Module):
    def __init__(self, target_size=128):
        super().__init__()
        self.target_size = target_size
        vgg = models.vgg16(
            pretrained=True
        ).features[:16].to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, fake, real):
        if self.target_size is not None:
            fake = F.interpolate(fake, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
            real = F.interpolate(real, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        fake_norm = (fake + 1) / 2
        real_norm = (real + 1) / 2
        fake_3c = fake_norm.repeat(1, 3, 1, 1)
        real_3c = real_norm.repeat(1, 3, 1, 1)
        fake_3c = (fake_3c - self.mean) / self.std
        real_3c = (real_3c - self.mean) / self.std
        with torch.no_grad():
            feat_real = self.vgg(real_3c)
        feat_fake = self.vgg(fake_3c)
        return F.l1_loss(feat_fake, feat_real)

class ContinuityLoss(nn.Module):
    def __init__(self, levels=4):
        super().__init__()
        self.levels = levels

    def _build_laplacian_pyramid(self, img, levels):
        pyramid = []
        current = img
        for _ in range(levels - 1):
            down = F.interpolate(current, scale_factor=0.5, mode="bilinear", align_corners=False)
            up = F.interpolate(down, size=current.shape[-2:], mode="bilinear", align_corners=False)
            pyramid.append(current - up)
            current = down
        pyramid.append(current)
        return pyramid

    def forward(self, fake, real):
        fake_pyr = self._build_laplacian_pyramid(fake, self.levels)
        real_pyr = self._build_laplacian_pyramid(real, self.levels)
        loss = 0.0
        for f, r in zip(fake_pyr, real_pyr):
            loss = loss + F.l1_loss(f, r)
        return loss / self.levels

def sobel_grad(x):
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)

def bbox_recon_focus_loss(fake, real, cond):
    mask = build_soft_bbox_mask(cond, h=fake.shape[-2], w=fake.shape[-1], blur_ks=11)
    inside = F.l1_loss(fake * mask, real * mask)
    outside = F.l1_loss(fake * (1.0 - mask), real * (1.0 - mask))
    return inside + 0.35 * outside

def bbox_recon_focus_loss_per_sample(fake, real, cond):
    mask = build_soft_bbox_mask(cond, h=fake.shape[-2], w=fake.shape[-1], blur_ks=11)
    diff = torch.abs(fake - real)
    eps = 1e-6
    inside_sum = (diff * mask).flatten(1).sum(dim=1)
    inside_area = mask.flatten(1).sum(dim=1) + eps
    outside = (1.0 - mask).clamp(0.0, 1.0)
    outside_sum = (diff * outside).flatten(1).sum(dim=1)
    outside_area = outside.flatten(1).sum(dim=1) + eps
    inside_mean = inside_sum / inside_area
    outside_mean = outside_sum / outside_area
    return inside_mean + 0.35 * outside_mean

def bbox_energy_focus_loss(fake, cond):
    mask = build_soft_bbox_mask(cond, h=fake.shape[-2], w=fake.shape[-1], blur_ks=9)
    energy = torch.abs(fake)
    eps = 1e-6
    inside_sum = (energy * mask).flatten(1).sum(dim=1)
    inside_area = mask.flatten(1).sum(dim=1) + eps
    outside_sum = (energy * (1.0 - mask)).flatten(1).sum(dim=1)
    outside_area = (1.0 - mask).flatten(1).sum(dim=1) + eps
    inside_mean = inside_sum / inside_area
    outside_mean = outside_sum / outside_area
    return F.relu(outside_mean - inside_mean * 0.8 + 0.015).mean()

def crack_edge_loss(fake, real, cond):
    crack_w = cond[:, 0:1].view(-1, 1, 1, 1)
    if torch.max(crack_w).item() <= 0:
        return fake.new_tensor(0.0)
    mask = build_soft_bbox_mask(cond, h=fake.shape[-2], w=fake.shape[-1], blur_ks=7)
    gf = sobel_grad(fake)
    gr = sobel_grad(real)
    weight = crack_w * mask
    num = torch.sum(torch.abs(gf - gr) * weight)
    den = torch.sum(weight) + 1e-6
    return num / den

def _highpass(x):
    return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

def background_noise_loss(fake, cond):
    mask = build_soft_bbox_mask(cond, h=fake.shape[-2], w=fake.shape[-1], blur_ks=11)
    bg = (1.0 - mask).clamp(0.0, 1.0)
    hp = torch.abs(_highpass(fake))
    dx = torch.abs(fake[:, :, :, 1:] - fake[:, :, :, :-1])
    dy = torch.abs(fake[:, :, 1:, :] - fake[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    tv = dx + dy
    loss_hp = (hp * bg).mean()
    loss_tv = (tv * bg).mean()
    return loss_hp + 0.35 * loss_tv

def defect_detail_loss(fake, real, cond):
    mask = build_soft_bbox_mask(cond, h=fake.shape[-2], w=fake.shape[-1], blur_ks=9)
    w = mask.clamp(0.0, 1.0)
    hp_f = _highpass(fake)
    hp_r = _highpass(real)
    e_f = sobel_grad(fake)
    e_r = sobel_grad(real)
    num1 = torch.sum(torch.abs(hp_f - hp_r) * w)
    num2 = torch.sum(torch.abs(e_f - e_r) * w)
    den = torch.sum(w) + 1e-6
    return num1 / den + 0.5 * (num2 / den)

def _masked_reduce_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.ndim == 4 and x.size(1) != 1:
        x = torch.mean(torch.abs(x), dim=1, keepdim=True)
    x = x * mask
    num = x.flatten(1).sum(dim=1)
    den = mask.flatten(1).sum(dim=1) + eps
    return num / den

def att_align_loss(att: torch.Tensor, cond: torch.Tensor, h: int, w: int, target: torch.Tensor = None, blur_ks: int = 3) -> torch.Tensor:
    if target is None:
        target = build_soft_bbox_mask(cond, h=h, w=w, blur_ks=blur_ks)
    att = att.clamp(1e-4, 1.0 - 1e-4)
    bce = F.binary_cross_entropy(att, target)
    eps = 1e-6
    inter = (att * target).flatten(1).sum(dim=1)
    denom = att.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1) + eps
    dice = 1.0 - (2.0 * inter + eps) / denom
    return bce + dice.mean()

def delta_map_align_loss(delta_feat: torch.Tensor, cond: torch.Tensor, h: int, w: int, target: torch.Tensor = None, blur_ks: int = 3) -> torch.Tensor:
    if target is None:
        target = build_soft_bbox_mask(cond, h=h, w=w, blur_ks=blur_ks)
    delta_heat = torch.mean(torch.abs(delta_feat), dim=1, keepdim=True)
    eps = 1e-6
    scale = delta_heat.mean(dim=(2, 3), keepdim=True) + eps
    delta_rel = (delta_heat / scale).clamp(0.0, 6.0)
    delta_prob = torch.sigmoid((delta_rel - 1.0) * 2.0)
    delta_prob = delta_prob.clamp(1e-4, 1.0 - 1e-4)
    bce = F.binary_cross_entropy(delta_prob, target)
    inter = (delta_prob * target).flatten(1).sum(dim=1)
    denom = delta_prob.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1) + eps
    dice = 1.0 - (2.0 * inter + eps) / denom
    return bce + dice.mean()

def delta_bbox_focus_loss(delta_feat: torch.Tensor, cond: torch.Tensor, h: int, w: int) -> torch.Tensor:
    mask = build_soft_bbox_mask(cond, h=h, w=w, blur_ks=1)
    inside = _masked_reduce_mean(delta_feat, mask)
    outside_mask = (1.0 - mask).clamp(0.0, 1.0)
    outside = _masked_reduce_mean(delta_feat, outside_mask)
    return F.relu(outside - inside * 0.8 + 0.01).mean()

def gain_floor_loss(gen: nn.Module, floor: float = 0.05) -> torch.Tensor:
    g1 = torch.clamp(getattr(gen, "clsea_gain"), 0.0, 1.0)
    g2 = torch.clamp(getattr(gen, "sldea_gain"), 0.0, 1.0)
    return F.relu(floor - g1) + F.relu(floor - g2)

def normalize_01_per_sample(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    b = x.size(0)
    flat = x.view(b, -1)
    minv = flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    maxv = flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    return (x - minv) / (maxv - minv + eps)

def feat_to_prob_1c(feat: torch.Tensor, k: float = 1.25, eps: float = 1e-6) -> torch.Tensor:
    heat = torch.mean(torch.abs(feat), dim=1, keepdim=True)
    mean = heat.mean(dim=(2, 3), keepdim=True)
    var = (heat - mean).pow(2).mean(dim=(2, 3), keepdim=True)
    std = torch.sqrt(var + eps)
    z = (heat - mean) / std
    return torch.sigmoid(k * z)

def jet_colormap(x01: torch.Tensor) -> torch.Tensor:
    x = x01.clamp(0.0, 1.0)
    r = (1.5 - torch.abs(4.0 * x - 3.0)).clamp(0.0, 1.0)
    g = (1.5 - torch.abs(4.0 * x - 2.0)).clamp(0.0, 1.0)
    b = (1.5 - torch.abs(4.0 * x - 1.0)).clamp(0.0, 1.0)
    return torch.cat([r, g, b], dim=1)

def feat_map_align_loss(feat: torch.Tensor, cond: torch.Tensor, target: torch.Tensor = None, blur_ks: int = 3) -> torch.Tensor:
    h, w = feat.shape[-2], feat.shape[-1]
    if target is None:
        target = build_soft_bbox_mask(cond, h=h, w=w, blur_ks=blur_ks)
    prob = feat_to_prob_1c(feat).clamp(1e-4, 1.0 - 1e-4)
    bce = F.binary_cross_entropy(prob, target)
    eps = 1e-6
    inter = (prob * target).flatten(1).sum(dim=1)
    denom = prob.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1) + eps
    dice = 1.0 - (2.0 * inter + eps) / denom
    return bce + dice.mean()

# ========================
# 模型组件
# ========================
class ResBlockUp(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_c, out_c, 3, 1, 1))
        self.skip = nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 1, 1, 0)) if in_c != out_c else nn.Identity()
        self.norm1 = nn.InstanceNorm2d(in_c, affine=True)
        self.norm2 = nn.InstanceNorm2d(out_c, affine=True)

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.conv1(F.leaky_relu(self.norm1(x_up), 0.2, inplace=True))
        h = self.conv2(F.leaky_relu(self.norm2(h), 0.2, inplace=True))
        return h + self.skip(x_up)

class MaskInjector(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        self.proj = nn.utils.spectral_norm(nn.Conv2d(1, out_c, 1, 1, 0))
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, mask):
        return x + torch.clamp(self.scale, 0.0, 1.0) * self.proj(mask)

class CLSEA(nn.Module):
    def __init__(self, channels, cond_dim=cond_dim, n_classes=n_classes):
        super().__init__()
        self.channels = channels

        # 四个方向：0°、45°、90°、135°
        self.num_directions = 4
        self.directions = [0, 45, 90, 135]

        # 四个可学习的方向卷积分支
        self.dir_convs = nn.ModuleList([
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            )
            for _ in range(self.num_directions)
        ])

        # 类别条件投影
        self.cond_proj = nn.Sequential(
            nn.Linear(n_classes, channels),
            nn.LeakyReLU(0.2, True),
            nn.Linear(channels, channels)
        )

        # 注意力融合
        self.att_fusion = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(channels * 2 + 1, channels, 1)
            ),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(
                nn.Conv2d(channels, 1, 1)
            ),
            nn.Sigmoid()
        )

        # 残差特征增强
        self.enhance = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(channels + 1, channels, 3, padding=1)
            ),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(
                nn.Conv2d(channels, channels, 3, padding=1)
            ),
            nn.Tanh()
        )

        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, mask, class_label):
        B, C, H, W = x.shape

        # 1. 四个方向分支的特征提取
        dir_feats = []
        for conv in self.dir_convs:
            feat = F.leaky_relu(conv(x), 0.2, True)
            dir_feats.append(feat)

        # [B, 4, C, H, W]
        dir_stack = torch.stack(dir_feats, dim=1)

        # 2. 结构能量图
        # 先在通道维度取最大值，再对四个分支求平均
        # [B, 1, H, W]
        dir_energy = torch.mean(
            torch.max(dir_stack, dim=2)[0],
            dim=1,
            keepdim=True
        )

        # 3. 类别条件特征
        cond_feat = self.cond_proj(class_label)
        cond_feat = cond_feat.view(B, C, 1, 1)
        cond_feat = cond_feat.expand(-1, -1, H, W)

        # 4. 方向能量与类别信息融合
        dir_energy_rep = dir_energy.repeat(1, C, 1, 1)

        att = self.att_fusion(
            torch.cat(
                [dir_energy_rep, cond_feat, mask],
                dim=1
            )
        )

        # 5. BBox 引导的注意力增强
        att = att * (0.1 + 0.9 * mask)
        att = torch.pow(att, 0.7)

        # 6. 残差特征增强
        residual = self.enhance(
            torch.cat([x, mask], dim=1)
        )

        delta = residual * att * mask

        # 7. 残差融合
        out = x + torch.clamp(
            self.alpha, 0.0, 1.0
        ) * delta

        return out, att

class SLDEA(nn.Module):
    def __init__(self, channels, cond_dim=cond_dim, n_classes=n_classes):
        super().__init__()
        self.channels = channels
        self.offset_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels + 1, channels, 3, padding=1)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.LeakyReLU(0.2, True),
        )
        self.offset_predictor = nn.Conv2d(channels, 18, 3, padding=1, bias=False)
        nn.init.zeros_(self.offset_predictor.weight)
        self.deform_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        nn.init.kaiming_normal_(self.deform_conv.weight, mode='fan_out', nonlinearity='relu')
        self.class_mod = nn.Sequential(
            nn.Linear(n_classes, channels),
            nn.Sigmoid()
        )
        self.gate_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels + 1, channels, 1)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(channels, 1, 1)),
            nn.Sigmoid()
        )
        self.beta = nn.Parameter(torch.tensor(0.15))

    def forward(self, x, mask, class_label):
        B, C, H, W = x.shape
        offset_input = torch.cat([x, mask], dim=1)
        offset_feat = self.offset_conv(offset_input)
        offset = self.offset_predictor(offset_feat)
        mask_deform = torch.ones(B, 9, H, W, device=x.device)
        aligned_x = deform_conv2d(x, offset, self.deform_conv.weight, mask=mask_deform, padding=1)
        class_weight = self.class_mod(class_label).view(B, C, 1, 1)
        continuity_feat = aligned_x * class_weight
        gate = self.gate_conv(torch.cat([x, mask], dim=1))
        gate = gate * mask
        delta = (continuity_feat - x) * gate
        out = x + torch.clamp(self.beta, 0.0, 1.0) * delta
        att = torch.mean(gate, dim=1, keepdim=True)
        return out, att

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, True),
        )
        self.fc = nn.Linear(latent_dim + 256, 512 * 8 * 8)
        self.up1 = ResBlockUp(512, 256)
        self.up2 = ResBlockUp(256, 128)
        self.up3 = ResBlockUp(128, 64)
        self.up4 = ResBlockUp(64, 32)
        self.up5 = ResBlockUp(32, 16)
        self.mask_inj16 = MaskInjector(256)
        self.mask_inj32 = MaskInjector(128)
        self.mask_inj64 = MaskInjector(64)
        self.mask_inj128 = MaskInjector(32)
        self.mask_inj256 = MaskInjector(16)

        self.clsea = CLSEA(256)
        self.sldea = SLDEA(64)
        self.clsea_gain = nn.Parameter(torch.tensor(0.35))
        self.sldea_gain = nn.Parameter(torch.tensor(0.15))

        self.to_img = nn.utils.spectral_norm(nn.Conv2d(16, 1, 3, 1, 1))
        self.out_scale = nn.Parameter(torch.tensor(0.8))

    def forward(self, z, cond, return_aux: bool = False):
        c = self.cond_emb(cond)
        x = self.fc(torch.cat([z, c], dim=1)).view(-1, 512, 8, 8)
        m16 = build_soft_bbox_mask(cond, h=16, w=16, blur_ks=5)
        m32 = build_soft_bbox_mask(cond, h=32, w=32, blur_ks=5)
        m64 = build_soft_bbox_mask(cond, h=64, w=64, blur_ks=7)
        m128 = build_soft_bbox_mask(cond, h=128, w=128, blur_ks=9)
        m256 = build_soft_bbox_mask(cond, h=256, w=256, blur_ks=11)
        class_label = cond[:, :n_classes]

        x = self.up1(x)
        x = self.mask_inj16(x, m16)
        x_clsea, att_clsea = self.clsea(x, m16, class_label)
        delta_clsea = x_clsea - x
        delta_clsea = delta_clsea * (0.02 + 0.98 * m16)
        x = x + torch.clamp(self.clsea_gain, 0.0, 1.0) * delta_clsea
        feat_after_clsea = x

        x = self.up2(x)
        x = self.mask_inj32(x, m32)
        x = self.up3(x)
        x = self.mask_inj64(x, m64)
        x_sldea, att_sldea = self.sldea(x, m64, class_label)
        delta_sldea = x_sldea - x
        delta_sldea = delta_sldea * (0.02 + 0.98 * m64)
        x = x + torch.clamp(self.sldea_gain, 0.0, 1.0) * delta_sldea
        feat_after_sldea = x

        x = self.up4(x)
        x = self.mask_inj128(x, m128)
        x = self.up5(x)
        x = self.mask_inj256(x, m256)
        x = self.to_img(F.leaky_relu(x, 0.2, True))
        img = torch.tanh(x) * torch.clamp(self.out_scale, 0.2, 1.0)

        if not return_aux:
            return img
        aux = {
            "att_clsea": att_clsea,
            "att_sldea": att_sldea,
            "delta_clsea": delta_clsea,
            "delta_sldea": delta_sldea,
            "feat_after_clsea": feat_after_clsea,
            "feat_after_sldea": feat_after_sldea,
        }
        return img, aux

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def d_block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, stride, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.from_img = d_block(2, 64)
        self.b1 = d_block(64, 128)
        self.b2 = d_block(128, 256)
        self.b3 = d_block(256, 512)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.fc = nn.utils.spectral_norm(nn.Linear(512, 1))
        self.cond_emb = nn.utils.spectral_norm(nn.Linear(cond_dim, 512))

    def forward(self, img, cond, bbox_mask):
        h = self.from_img(torch.cat([img, bbox_mask], dim=1))
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        feats = [h]
        h = self.conv(h)
        h = F.leaky_relu(h, 0.2, True)
        h_pool = torch.mean(h, dim=(2, 3))
        out_uncond = self.fc(h_pool)
        c = self.cond_emb(cond)
        out_proj = torch.sum(h_pool * c, dim=1, keepdim=True)
        out = out_uncond + out_proj
        return out, feats

# ========================
# 损失函数（Hinge）
# ========================
def d_hinge_loss(d_real, d_fake):
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()

def g_hinge_loss(d_fake):
    return -d_fake.mean()

def r1_regularize(d_out, real_img):
    grad_real = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=real_img,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad_real.pow(2).reshape(real_img.size(0), -1).sum(1).mean()

# ========================
# 断点保存与恢复函数
# ========================
def save_checkpoint(epoch, step, model_G, model_D, opt_G, opt_D, sched_G, sched_D, scaler_G, scaler_D, fixed_z):
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_G_state_dict': model_G.state_dict(),
        'model_D_state_dict': model_D.state_dict(),
        'optimizer_G_state_dict': opt_G.state_dict(),
        'optimizer_D_state_dict': opt_D.state_dict(),
        'scheduler_G_state_dict': sched_G.state_dict(),
        'scheduler_D_state_dict': sched_D.state_dict(),
        'scaler_G_state_dict': scaler_G.state_dict() if scaler_G else None,
        'scaler_D_state_dict': scaler_D.state_dict() if scaler_D else None,
        'fixed_z': fixed_z,
    }, CHECKPOINT_PATH)
    print(f"Checkpoint saved at epoch {epoch}, step {step}")

def load_checkpoint(model_G, model_D, opt_G, opt_D, sched_G, sched_D, scaler_G, scaler_D, fixed_z):
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model_G.load_state_dict(checkpoint['model_G_state_dict'])
        model_D.load_state_dict(checkpoint['model_D_state_dict'])
        opt_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        opt_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        sched_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        sched_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        if scaler_G and checkpoint['scaler_G_state_dict']:
            scaler_G.load_state_dict(checkpoint['scaler_G_state_dict'])
        if scaler_D and checkpoint['scaler_D_state_dict']:
            scaler_D.load_state_dict(checkpoint['scaler_D_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch继续
        start_step = checkpoint['step']
        # 恢复固定噪声
        fixed_z.copy_(checkpoint['fixed_z'])
        print(f"Resuming from epoch {start_epoch}, step {start_step}")
        return start_epoch, start_step
    else:
        print("No checkpoint found, starting from scratch.")
        return 0, 0

# ========================
# 训练主循环
# ========================
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = BScanDataset("split_dataset/train", transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    gen = Generator().to(device)
    dis = Discriminator().to(device)
    g_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in dis.parameters() if p.requires_grad)
    total_params = g_params + d_params

    print(f"Generator Params: {g_params / 1e6:.2f} M")
    print(f"Discriminator Params: {d_params / 1e6:.2f} M")
    print(f"Total Params: {total_params / 1e6:.2f} M")

    per_loss = PerceptualLoss(target_size=perceptual_imgsz)
    cont_loss = ContinuityLoss(levels=4)

    opt_G = optim.Adam(list(gen.parameters()), lr=lr_g, betas=betas)
    opt_D = optim.Adam(dis.parameters(), lr=lr_d, betas=betas)

    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=num_epochs)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=num_epochs)

    fixed_real, fixed_conds = build_class_balanced_preview(dataset)
    fixed_real = fixed_real.to(device, non_blocking=True)
    fixed_conds = fixed_conds.to(device, non_blocking=True)
    fixed_z = torch.randn(fixed_conds.size(0), latent_dim, device=device)

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler_D = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    scaler_G = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # 加载断点
    start_epoch, global_step = load_checkpoint(gen, dis, opt_G, opt_D, sched_G, sched_D, scaler_G, scaler_D, fixed_z)


    def autocast_ge_ctx():
        if amp_enabled:
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()


    def autocast_dis_ctx():
        if amp_enabled:
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    print("开始训练：模块重设计版（方向感知弱信号增强 + 可变形连续性修复）...")
    print(f"数据集大小: {len(dataset)}")

    # 恢复训练状态变量
    step = global_step
    for epoch in range(start_epoch, num_epochs):
        adv_ratio = 0.0 if epoch < warmup_epochs else min(1.0, float(epoch - warmup_epochs + 1) / float(max(1, adv_ramp_epochs)))
        opt_D.zero_grad(set_to_none=True)
        opt_G.zero_grad(set_to_none=True)
        for i, (real_imgs, conds) in enumerate(dataloader):
            real_imgs = real_imgs.to(device, non_blocking=True)
            conds = conds.to(device, non_blocking=True)
            mask256 = build_soft_bbox_mask(conds, h=256, w=256, blur_ks=11).to(device)
            is_update_step = ((i + 1) % grad_accum_steps == 0) or ((i + 1) == len(dataloader))

            is_crack = conds[:, 0:1]
            with torch.no_grad():
                edge_target_16 = build_edge_focused_target(real_imgs, conds, h=16, w=16, blur_ks=3, edge_blur=3)
                edge_target_64 = build_edge_focused_target(real_imgs, conds, h=64, w=64, blur_ks=3, edge_blur=5)
                edge_target_256 = build_edge_focused_target(real_imgs, conds, h=256, w=256, blur_ks=5, edge_blur=7)

            # ---- Train D ----
            if adv_ratio > 0.0:
                with torch.no_grad():
                    with autocast_ge_ctx():
                        z = torch.randn(real_imgs.size(0), latent_dim, device=device)
                        fake_imgs = gen(z, conds)

                real_imgs.requires_grad_(True)
                with autocast_dis_ctx():
                    real_fp32 = real_imgs.float()
                    fake_fp32 = fake_imgs.detach().float()
                    d_real, _ = dis(real_fp32, conds, mask256)
                    d_fake, _ = dis(fake_fp32, conds, mask256)
                    loss_D = d_hinge_loss(d_real, d_fake)

                if step % r1_every == 0:
                    with autocast_dis_ctx():
                        r1 = r1_regularize(d_real, real_fp32)
                    loss_D = loss_D + 0.5 * r1_gamma * r1

                if torch.isfinite(loss_D).all():
                    scaler_D.scale(loss_D / grad_accum_steps).backward()
                    if is_update_step:
                        scaler_D.unscale_(opt_D)
                        torch.nn.utils.clip_grad_norm_(dis.parameters(), max_norm=max_grad_norm_d)
                        scaler_D.step(opt_D)
                        scaler_D.update()
                        opt_D.zero_grad(set_to_none=True)
                else:
                    opt_D.zero_grad(set_to_none=True)
                    scaler_D.update()
                real_imgs.requires_grad_(False)
            else:
                loss_D = real_imgs.new_tensor(0.0)

            # ---- Train G ----
            with autocast_ge_ctx():
                if int(best_of_k) <= 1:
                    z = torch.randn(real_imgs.size(0), latent_dim, device=device)
                    gen_imgs, aux = gen(z, conds, return_aux=True)
                else:
                    k = int(best_of_k)
                    z_all = torch.randn(k, real_imgs.size(0), latent_dim, device=device)
                    gen_list = []
                    for t in range(k):
                        gen_list.append(gen(z_all[t], conds, return_aux=False))
                    gen_stack = torch.stack(gen_list, dim=0)
                    with autocast_dis_ctx():
                        rf = real_imgs.float()
                        sel_losses = []
                        for t in range(k):
                            sel_losses.append(bbox_recon_focus_loss_per_sample(gen_stack[t].float(), rf, conds))
                        sel_losses = torch.stack(sel_losses, dim=0)
                        best_idx = torch.argmin(sel_losses, dim=0)
                    ar = torch.arange(real_imgs.size(0), device=device)
                    z_sel = z_all[best_idx, ar]
                    gen_imgs, aux = gen(z_sel, conds, return_aux=True)

                if adv_ratio > 0.0:
                    with autocast_dis_ctx():
                        gen_fp32 = gen_imgs.float()
                        real_fp32_2 = real_imgs.float()
                        d_fake_g, f_fake = dis(gen_fp32, conds, mask256)
                        with torch.no_grad():
                            _, f_real = dis(real_fp32_2, conds, mask256)

                    loss_adv = g_hinge_loss(d_fake_g) * adv_ratio
                    loss_fm = 0.0
                    for a, b in zip(f_fake, f_real):
                        loss_fm = loss_fm + F.l1_loss(a, b)
                    loss_fm = loss_fm * lambda_fm * adv_ratio
                else:
                    loss_adv = gen_imgs.new_tensor(0.0)
                    loss_fm = gen_imgs.new_tensor(0.0)

                if lambda_perceptual > 0:
                    with autocast_dis_ctx():
                        loss_per = per_loss(gen_imgs.float(), real_imgs.float()) * lambda_perceptual
                else:
                    loss_per = gen_imgs.new_tensor(0.0)

                with autocast_dis_ctx():
                    gf = gen_imgs.float()
                    rf = real_imgs.float()
                    loss_cont = cont_loss(gf, rf) * lambda_continuity
                    loss_bbox = bbox_recon_focus_loss(gf, rf, conds) * lambda_bbox
                    loss_focus = bbox_energy_focus_loss(gf, conds) * lambda_focus
                    loss_edge = crack_edge_loss(gf, rf, conds) * lambda_edge
                    loss_bg = background_noise_loss(gf, conds) * lambda_bg
                    loss_detail = defect_detail_loss(gf, rf, conds) * lambda_detail

                    att_c = aux["att_clsea"].float()
                    att_s = aux["att_sldea"].float()
                    delta_c = aux["delta_clsea"].float()
                    delta_s = aux["delta_sldea"].float()

                    loss_att_c = att_align_loss(att_c, conds, h=att_c.shape[-2], w=att_c.shape[-1],
                                                target=edge_target_16, blur_ks=3)
                    loss_att_s = att_align_loss(att_s, conds, h=att_s.shape[-2], w=att_s.shape[-1],
                                                target=edge_target_64, blur_ks=3)
                    loss_att = (loss_att_c + loss_att_s) * lambda_att

                    loss_delta = (delta_bbox_focus_loss(delta_c, conds, h=delta_c.shape[-2], w=delta_c.shape[-1])
                                  + delta_bbox_focus_loss(delta_s, conds, h=delta_s.shape[-2], w=delta_s.shape[-1])) * lambda_delta

                    loss_dm_c = delta_map_align_loss(delta_c, conds, h=delta_c.shape[-2], w=delta_c.shape[-1],
                                                     target=edge_target_16, blur_ks=3)
                    loss_dm_s = delta_map_align_loss(delta_s, conds, h=delta_s.shape[-2], w=delta_s.shape[-1],
                                                     target=edge_target_64, blur_ks=3)
                    loss_delta_map = (loss_dm_c + loss_dm_s) * lambda_delta_map

                    loss_fm_c = feat_map_align_loss(aux["feat_after_clsea"].float(), conds,
                                                    target=edge_target_16, blur_ks=3)
                    loss_fm_s = feat_map_align_loss(aux["feat_after_sldea"].float(), conds,
                                                    target=edge_target_64, blur_ks=3)
                    loss_feat_map = (loss_fm_c + loss_fm_s) * lambda_feat_map

                    loss_gain = gain_floor_loss(gen) * lambda_gain_floor

                boost_factor = 1.0 + (lambda_crack_boost - 1.0) * is_crack.float().mean()
                loss_att = loss_att * boost_factor
                loss_delta = loss_delta * boost_factor
                loss_delta_map = loss_delta_map * boost_factor
                loss_feat_map = loss_feat_map * boost_factor

                loss_G = (loss_adv + loss_fm + loss_per + loss_cont + loss_bbox + loss_focus +
                          loss_edge + loss_bg + loss_detail + loss_att + loss_delta +
                          loss_delta_map + loss_feat_map + loss_gain)

            if torch.isfinite(loss_G).all():
                scaler_G.scale(loss_G / grad_accum_steps).backward()
                if is_update_step:
                    scaler_G.unscale_(opt_G)
                    torch.nn.utils.clip_grad_norm_(list(gen.parameters()), max_norm=max_grad_norm_g)
                    scaler_G.step(opt_G)
                    scaler_G.update()
                    opt_G.zero_grad(set_to_none=True)
            else:
                opt_G.zero_grad(set_to_none=True)
                scaler_G.update()

            step += 1

        sched_G.step()
        sched_D.step()

        # 每个 epoch 结束后保存断点
        save_checkpoint(epoch, step, gen, dis, opt_G, opt_D, sched_G, sched_D, scaler_G, scaler_D, fixed_z)

        if epoch % 5 == 0:
            preview_n = fixed_conds.size(0)
            print(
                f"Epoch {epoch:3d} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f} | "
                f"Adv: {loss_adv.item():.4f} | FM: {loss_fm.item():.4f} | "
                f"Cont: {loss_cont.item():.4f} | BBox: {loss_bbox.item():.4f} | Ramp: {adv_ratio:.2f} | "
                f"Focus: {loss_focus.item():.4f} | Edge: {loss_edge.item():.4f} | Bg: {loss_bg.item():.4f} | Detail: {loss_detail.item():.4f} | "
                f"Att: {loss_att.item():.4f} | Delta: {loss_delta.item():.4f} | DMap: {loss_delta_map.item():.4f} | FMap: {loss_feat_map.item():.4f} | Gain: {loss_gain.item():.4f}"
            )
            with torch.no_grad():
                gen.eval()
                sample = gen(fixed_z, fixed_conds)
                preview = torch.cat([fixed_real, sample], dim=0)
                save_image(preview, f"{RESULTS_DIR}/epoch_{epoch}.png", nrow=preview_n, normalize=True)

                _, aux2 = gen(fixed_z, fixed_conds, return_aux=True)
                mask_fix = build_soft_bbox_mask(fixed_conds, h=256, w=256, blur_ks=11).to(fixed_real.device)
                feat_c = aux2["feat_after_clsea"]
                feat_s = aux2["feat_after_sldea"]
                h_c = F.interpolate(feat_to_prob_1c(feat_c), size=img_size, mode="bilinear", align_corners=False)
                h_s = F.interpolate(feat_to_prob_1c(feat_s), size=img_size, mode="bilinear", align_corners=False)
                mask_rgb = jet_colormap(mask_fix)
                hc_rgb = jet_colormap(h_c)
                hs_rgb = jet_colormap(h_s)
                panel = torch.cat([mask_rgb, hc_rgb, hs_rgb], dim=0)
                save_image(panel, f"{RESULTS_DIR}/epoch_{epoch}_bbox_featmaps.png", nrow=preview_n, normalize=False)

                if adv_ratio > 0.0:
                    prior_z = torch.randn(preview_n, latent_dim, device=device)
                    prior_sample = gen(prior_z, fixed_conds)
                    save_image(prior_sample, f"{RESULTS_DIR}/epoch_{epoch}_prior.png", nrow=preview_n, normalize=True)
                gen.train()

    torch.save(gen.state_dict(), "generator_enhanced3.pth")
    print("训练完成，模型已保存为 generator_enhanced3.pth")