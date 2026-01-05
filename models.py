import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config


class EfficientAttention(nn.Module):
    """高效注意力机制 - 线性复杂度版本"""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 线性注意力计算
        k = F.softmax(k, dim=-2)  # 在序列维度softmax
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        attn_out = torch.einsum('bhnd,bhde->bhne', q, context)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.out_proj(attn_out))


class SpatialGroupedAttention(nn.Module):
    """空间分组注意力机制 - 优化版"""

    def __init__(self, dim, num_heads=8, group_size=5, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # 生成QKV
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 动态分组
        if T <= self.group_size:
            # 序列较短时直接计算全局注意力
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        else:
            # 分组处理
            num_groups = (T + self.group_size - 1) // self.group_size
            pad_len = num_groups * self.group_size - T

            if pad_len > 0:
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))

            # 重塑为分组形式
            q = q.view(B, self.num_heads, num_groups, self.group_size, self.head_dim)
            k = k.view(B, self.num_heads, num_groups, self.group_size, self.head_dim)
            v = v.view(B, self.num_heads, num_groups, self.group_size, self.head_dim)

            # 计算组内注意力
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # 应用注意力并恢复形状
            x = (attn @ v).transpose(2, 3).contiguous()
            x = x.view(B, self.num_heads, num_groups * self.group_size, self.head_dim)

            if pad_len > 0:
                x = x[:, :, :T, :]

            x = x.transpose(1, 2).contiguous().view(B, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiScaleAttention(nn.Module):
    """多尺度注意力机制"""

    def __init__(self, dim, num_heads=8, scales=[1, 2], dropout=0.1):
        super().__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            EfficientAttention(dim, num_heads, dropout) for _ in scales
        ])
        self.fusion = nn.Linear(dim * len(scales), dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        outputs = []

        for scale, attn in zip(self.scales, self.attentions):
            if scale > 1 and T > scale * 2:
                # 多尺度处理
                x_down = F.avg_pool1d(x.transpose(1, 2), kernel_size=scale, stride=scale)
                x_down = x_down.transpose(1, 2)
                attn_out = attn(x_down)
                # 上采样
                attn_out = F.interpolate(attn_out.transpose(1, 2), size=T,
                                         mode='linear', align_corners=False)
                attn_out = attn_out.transpose(1, 2)
            else:
                attn_out = attn(x)
            outputs.append(attn_out)

        fused = self.fusion(torch.cat(outputs, dim=-1))
        return self.dropout(fused)


class HybridTransformerBlock(nn.Module):
    """混合Transformer块 - 核心创新"""

    def __init__(self, dim, num_heads=8, attention_type="hybrid", group_size=5, dropout=0.1):
        super().__init__()
        self.attention_type = attention_type

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 根据类型选择注意力机制
        if attention_type == "group":
            self.attn = SpatialGroupedAttention(dim, num_heads, group_size, dropout)
        elif attention_type == "efficient":
            self.attn = EfficientAttention(dim, num_heads, dropout)
        elif attention_type == "multiscale":
            self.attn = MultiScaleAttention(dim, num_heads, dropout=dropout)
        elif attention_type == "hybrid":
            # 并行混合注意力
            self.group_attn = SpatialGroupedAttention(dim, num_heads // 2, group_size, dropout)
            self.efficient_attn = EfficientAttention(dim, num_heads // 2, dropout)
            self.attention_weights = nn.Parameter(torch.ones(2))
            self.fusion = nn.Linear(dim * 2, dim)
        else:
            raise ValueError(f"不支持的注意力类型: {attention_type}")

        self.attention_type = attention_type

        # 轻量化MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),  # 减少扩展比例
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 注意力部分
        if self.attention_type == "hybrid":
            group_out = self.group_attn(self.norm1(x))
            efficient_out = self.efficient_attn(self.norm1(x))
            weights = F.softmax(self.attention_weights, dim=0)
            attn_out = weights[0] * group_out + weights[1] * efficient_out
            # 可选：使用融合层而不是加权平均
            # attn_out = self.fusion(torch.cat([group_out, efficient_out], dim=-1))
        else:
            attn_out = self.attn(self.norm1(x))

        x = x + self.dropout(attn_out)

        # MLP部分
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_out)

        return x


class LightweightVisionTransformer(nn.Module):
    """轻量化Vision Transformer - 增强版"""

    def __init__(self, input_dim=1592, seq_length=10, num_classes=29,
                 embed_dim=256, depth=6, num_heads=8, dropout=0.1,
                 attention_config="alternating"):
        super().__init__()

        self.seq_length = seq_length
        self.attention_config = attention_config

        # 更高效的特征投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_length, embed_dim))

        # 动态构建Transformer块
        self.blocks = self._build_blocks(embed_dim, num_heads, depth, dropout)

        # 归一化层
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        self._init_weights()

    def _build_blocks(self, embed_dim, num_heads, depth, dropout):
        """根据配置构建Transformer块"""
        blocks = nn.ModuleList()
        attention_types = []

        if self.attention_config == "alternating":
            # 交替使用不同注意力机制
            for i in range(depth):
                if i % 3 == 0:
                    attn_type = "group"
                elif i % 3 == 1:
                    attn_type = "efficient"
                else:
                    attn_type = "multiscale"
                attention_types.append(attn_type)

        elif self.attention_config == "hybrid":
            # 所有块都使用混合注意力
            attention_types = ["hybrid"] * depth
        else:
            # 单一注意力类型
            attention_types = [self.attention_config] * depth

        print(f"注意力配置: {attention_types}")

        for attn_type in attention_types:
            blocks.append(
                HybridTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    attention_type=attn_type,
                    dropout=dropout
                )
            )

        return blocks

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)
        x = x + self.pos_embed

        # 通过所有Transformer块
        for blk in self.blocks:
            x = blk(x)

        # 归一化和分类
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.head(x)

        return x


# 兼容原有代码的创建函数
def create_model(num_classes, model_type="lightweight"):
    """创建模型实例 - 增强版"""
    if model_type == "lightweight":
        model = LightweightVisionTransformer(
            input_dim=Config.INPUT_DIM,
            seq_length=Config.SEQ_LENGTH,
            num_classes=num_classes,
            embed_dim=Config.EMBED_DIM,
            depth=Config.DEPTH,
            num_heads=Config.NUM_HEADS,
            dropout=Config.DROPOUT,
            attention_config=Config.ATTENTION_CONFIG
        )
    else:
        # 其他模型类型...
        model = LightweightVisionTransformer(
            input_dim=Config.INPUT_DIM,
            seq_length=Config.SEQ_LENGTH,
            num_classes=num_classes,
            embed_dim=Config.EMBED_DIM,
            depth=Config.DEPTH,
            num_heads=Config.NUM_HEADS,
            dropout=Config.DROPOUT
        )

    return model.to(Config.DEVICE)


if __name__ == "__main__":
    # 测试增强模型
    model = create_model(num_classes=29)

    with torch.no_grad():
        test_input = torch.randn(2, 10, 1592).to(Config.DEVICE)
        test_output = model(test_input)
        print(f"增强模型测试通过")
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {test_output.shape}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")