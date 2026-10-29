import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.ops import deform_conv2d
import xml.etree.ElementTree as ET
from PIL import Image

# ---------- 设备 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------- 超参数（必须与训练一致）----------
latent_dim = 256
img_size = (256, 256)
n_classes = 4
class_names = ["DT_CRACK", "DT_GAP", "DT_LACUNAS", "DT_SUBSIDENCE"]
class_to_idx = {name: i for i, name in enumerate(class_names)}
cond_dim = n_classes + 4          # 8

# 权重路径 & 测试数据路径
model_path = "generator_enhanced3.pth"
test_xml_dir = "test_new_label_sub"
output_dir = "generated_test_images_enhanced3_sub"
os.makedirs(output_dir, exist_ok=True)

# ================== 与 SA-CGAN.py 训练结构逐字一致的网络定义 ==================

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

# ================== 加载生成器 ==================
generator = Generator().to(device)
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"找不到生成器权重: {model_path}。请检查 model_path 配置。")

# 支持直接保存的 generator.state_dict() 和训练中保存的完整 checkpoint。
checkpoint = torch.load(model_path, map_location=device, weights_only=True)
if isinstance(checkpoint, dict) and 'model_G_state_dict' in checkpoint:
    state_dict = checkpoint['model_G_state_dict']
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
# strict=True 用于确保每一层参数均与训练网络完全匹配。
generator.load_state_dict(state_dict, strict=True)
generator.eval()
print("✅ Generator loaded successfully (strict=True).")

# ================== XML 解析（bbox 归一化与训练完全一致） ==================
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []

    # 读取原图尺寸
    size_node = root.find('size')
    if size_node is not None:
        orig_w = float(size_node.find('width').text)
        orig_h = float(size_node.find('height').text)
    else:
        xml_dir = os.path.dirname(xml_path)
        base = os.path.splitext(os.path.basename(xml_path))[0]
        found = False
        for ext in ['.jpg', '.png', '.jpeg']:
            cand = os.path.join(xml_dir, base + ext)
            if os.path.exists(cand):
                with Image.open(cand) as im:
                    orig_w, orig_h = im.size
                found = True
                break
        if not found:
            orig_w, orig_h = 256, 256

    orig_w = float(max(1.0, orig_w))
    orig_h = float(max(1.0, orig_h))

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in class_to_idx:
            print(f"⚠️ Unknown class: {name} in {xml_path}")
            continue

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        x_norm = xmin / orig_w
        y_norm = ymin / orig_h
        w_norm = (xmax - xmin) / orig_w
        h_norm = (ymax - ymin) / orig_h

        objects.append({
            'class_name': name,
            'class_id': class_to_idx[name],
            'bbox_norm': (x_norm, y_norm, w_norm, h_norm)
        })
    return objects

if not os.path.isdir(test_xml_dir):
    raise FileNotFoundError(f"测试 XML 文件夹不存在: {test_xml_dir}")

# ================== 遍历 XML 并生成图像 ==================
for class_dir in os.listdir(test_xml_dir):
    class_path = os.path.join(test_xml_dir, class_dir)
    if not os.path.isdir(class_path):
        continue
    for xml_file in os.listdir(class_path):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(class_path, xml_file)
        try:
            objects = parse_xml(xml_path)
            base_name = os.path.splitext(xml_file)[0]

            for idx, obj in enumerate(objects):
                # 构造条件向量
                condition = torch.zeros(cond_dim, dtype=torch.float32)
                condition[obj['class_id']] = 1.0
                condition[n_classes:] = torch.tensor(obj['bbox_norm'], dtype=torch.float32)

                z = torch.randn(1, latent_dim, device=device)
                condition = condition.unsqueeze(0).to(device)

                with torch.no_grad():
                    gen_img = generator(z, condition)  # (1, 1, 256, 256)

                save_name = f"{base_name}_obj{idx}_{obj['class_name']}.png"
                save_path = os.path.join(output_dir, save_name)
                save_image(gen_img, save_path, nrow=1, normalize=True)
                print(f"✅ Saved: {save_path}")

        except Exception as e:
            print(f"❌ Error processing {xml_path}: {e}")

print("\n🎉 All test images generated using redesigned CLSEA & SLDEA model!")