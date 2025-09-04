import math
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import torchvision
from torchvision import transforms, datasets
from torch import optim
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# 确保目录存在
os.makedirs('images', exist_ok=True)
os.makedirs('data', exist_ok=True)  # 如果数据集不存在请提前创建


class Config(object):
    data_path = 'data/'  # 数据集路径
    image_size = 96  # 图像尺寸
    batch_size = 32  # 批大小
    epochs = 1  # 训练轮数
    lr1 = 2e-3  # 生成器学习率
    lr2 = 2e-4  # 判别器学习率
    beta1 = 0.5  # Adam优化器参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择设备
    nz = 100  # 噪声向量维度
    ngf = 64  # 生成器特征图基数
    ndf = 64  # 判别器特征图基数
    save_path = './images'  # 图像保存路径
    generator_path = './generator.pkl'  # 生成器保存路径
    discriminator_path = './discriminator.pkl'  # 判别器保存路径
    gen_img = 'result.png'  # 输出图像文件名
    gen_num = 64  # 生成图像数量
    gen_search_num = 10000  # 候选图像数量
    gen_mean = 0  # 噪声均值
    gen_std = 1  # 噪声标准差


config = Config()  # 实例化配置
print(f"Using device: {config.device}")

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(config.image_size),
    transforms.CenterCrop(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载数据集
train_dataset = datasets.ImageFolder(root=config.data_path, transform=data_transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True
)

print(f'Using {len(train_dataset)} images for training.')


# 定义生成器
class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        ngf = config.ngf
        self.model = nn.Sequential(
            nn.ConvTranspose2d(config.nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        ndf = config.ndf
        self.model = nn.Sequential(
            nn.Conv2d(3, ndf, 5, 3, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.model(x).view(-1)


# 初始化模型
generator = Generator(config).to(config.device)
discriminator = Discriminator(config).to(config.device)

# 优化器
optimizer_g = optim.Adam(generator.parameters(), lr=config.lr1, betas=(config.beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=config.lr2, betas=(config.beta1, 0.999))

# 固定噪声用于生成示例图像
fixed_noise = torch.randn(config.batch_size, config.nz, 1, 1, device=config.device)

# 训练循环
d_losses = []
g_losses = []

for epoch in range(config.epochs):
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}')

    for i, (real_imgs, _) in enumerate(progress_bar):
        # 将真实图像移动到设备
        real_imgs = real_imgs.to(config.device)
        batch_size = real_imgs.size(0)

        # ==============================
        #  训练判别器
        # ==============================
        optimizer_d.zero_grad()

        # 真实图像的损失
        real_preds = discriminator(real_imgs)
        d_loss_real = F.relu(1.0 - real_preds).mean()

        # 生成假图像
        noise = torch.randn(batch_size, config.nz, 1, 1, device=config.device)
        fake_imgs = generator(noise).detach()  # 分离梯度，不更新生成器

        # 假图像的损失
        fake_preds = discriminator(fake_imgs)
        d_loss_fake = F.relu(1.0 + fake_preds).mean()

        # 判别器总损失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # ==============================
        #  训练生成器
        # ==============================
        optimizer_g.zero_grad()

        # 生成新假图像
        noise = torch.randn(batch_size, config.nz, 1, 1, device=config.device)
        fake_imgs = generator(noise)

        # 生成器损失 - 让判别器认为假图像是真实的
        fake_preds = discriminator(fake_imgs)
        g_loss = -fake_preds.mean()

        g_loss.backward()
        optimizer_g.step()

        # 记录损失
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        # 更新进度条
        progress_bar.set_postfix({
            'D_loss': f'{d_loss.item():.4f}',
            'G_loss': f'{g_loss.item():.4f}'
        })

    # 每个epoch结束后保存示例图像
    with torch.no_grad():
        sample_imgs = generator(fixed_noise)
        save_path = os.path.join(config.save_path, f'epoch_{epoch + 1}.png')
        torchvision.utils.save_image(
            sample_imgs,
            save_path,
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )

    # 每5个epoch保存一次模型
    if (epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(), f'generator_epoch_{epoch + 1}.pth')
        torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch + 1}.pth')

# 训练完成后保存最终模型
torch.save(generator.state_dict(), config.generator_path)
torch.save(discriminator.state_dict(), config.discriminator_path)

# 生成最终结果图像
print("Generating final result images...")
generator.eval()
with torch.no_grad():
    # 生成候选图像
    noises = torch.randn(config.gen_search_num, config.nz, 1, 1, device=config.device)
    fake_imgs = generator(noises)

    # 评分并选择最佳图像
    scores = discriminator(fake_imgs)
    _, best_indices = torch.topk(scores, config.gen_num)
    best_imgs = fake_imgs[best_indices]

    # 保存结果
    torchvision.utils.save_image(
        best_imgs,
        config.gen_img,
        nrow=8,
        normalize=True,
        value_range=(-1, 1)
    )

print("Training completed!")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss History')
plt.savefig('loss_curve.png')
plt.show()