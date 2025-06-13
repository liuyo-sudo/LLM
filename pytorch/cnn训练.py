import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.jit
import torch.onnx
import os

# 禁用 libuv（解决之前的错误）
os.environ["USE_LIBUV"] = "0"

# 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = "nccl" if torch.cuda.is_available() and torch.cuda.nccl.is_available(torch.cuda.get_device_properties(rank)) else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 训练函数
def train(rank, world_size, epochs=3):
    setup(rank, world_size)

    # 数据处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=32, sampler=sampler, num_workers=0)

    # 初始化模型、损失函数、优化器和混合精度
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # TensorBoard
    writer = SummaryWriter() if rank == 0 else None

    # 检查点加载
    checkpoint_path = "checkpoint.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path) and rank == 0:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从第 {start_epoch} 轮恢复训练")
    dist.barrier()

    # 训练循环
    for epoch in range(start_epoch, epochs):
        model.train()
        sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if i % 50 == 49 and rank == 0:
                avg_loss = running_loss / 50
                print(f"[轮次 {epoch+1}, 批次 {i+1}] 损失: {avg_loss:.3f}")
                writer.add_scalar("Loss/train", avg_loss, epoch * len(trainloader) + i)
                running_loss = 0.0

        # 保存检查点
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, checkpoint_path)

    # 可视化（修复部分）
    if rank == 0:
        images, _ = next(iter(trainloader))
        # 确保图像张量形状为 (C, H, W)，添加通道维度
        sample_image = images[0]  # Shape: (1, 28, 28)
        if sample_image.dim() == 3 and sample_image.shape[0] == 1:
            sample_image = sample_image.squeeze(0)  # Shape: (28, 28)
            sample_image = sample_image.unsqueeze(0)  # Shape: (1, 28, 28)
        writer.add_image("MNIST Sample", sample_image, dataformats="CHW")
        writer.close()

    # TorchScript
    if rank == 0:
        model.eval()
        example_input = torch.randn(1, 1, 28, 28).to(device)
        traced_model = torch.jit.trace(model.module, example_input)
        traced_model.save("model.pt")

    # ONNX
    if rank == 0:
        torch.onnx.export(model.module, example_input, "model.onnx",
                          input_names=["input"], output_names=["output"], opset_version=11)

    cleanup()

# 主函数
def main():
    tensor = torch.randn(2, 3)
    print("示例张量:\n", tensor)
    print("转置后张量:\n", torch.transpose(tensor, 0, 1))
    print("张量和:", torch.sum(tensor))

    world_size = torch.cuda.device_count() or 1
    if world_size < 1:
        print("无 GPU 可用，使用 CPU")
        world_size = 1
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()