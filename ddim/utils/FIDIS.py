import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from torchvision import datasets, transforms, models
from pytorch_fid import fid_score
from scipy.stats import entropy
from PIL import Image


class FID_and_IS:
    """
    FID和IS计算器整合类
    统一将real图片放在../data/realimg下，fake图片放在tmp_dir下
    """
    
    # 类级别的共享real图片目录
    _shared_real_dir = "../data/realimg"
    _shared_real_prepared = False
    
    def __init__(self, device, model_name=None, real_batch_size=100, tmp_dir="./tmp_fid_is", is_splits=10, con_model=False):
        """
        初始化FID和IS计算器
        
        Parameters:
        -----------
        device : str
            计算设备 ('cuda' 或 'cpu')
        model_name : str, optional
            模型名称，用于区分不同模型的fake图片
        real_batch_size : int
            处理图片的批次大小
        tmp_dir : str
            临时文件目录（存放fake图片）
        is_splits : int
            计算IS时的分割数量，用于计算标准差
        con_model : bool
            是否为条件模型，如果为True则生成随机标签
        """
        self.device = device
        self.model_name = model_name or "default"
        self.real_batch_size = real_batch_size
        self.tmp_dir = tmp_dir
        self.is_splits = is_splits
        self.con_model = con_model  # 新增条件模型标志
        
        # 统一的real图片目录
        self.real_dir = FID_and_IS._shared_real_dir
        
        # fake图片直接放在tmp_dir下，按模型名区分
        if self.model_name == "default":
            self.fake_dir = self.tmp_dir
        else:
            self.fake_dir = os.path.join(self.tmp_dir, self.model_name)
        
        # 创建必要的目录
        os.makedirs(self.real_dir, exist_ok=True)
        os.makedirs(self.fake_dir, exist_ok=True)
        
        # 为IS计算加载Inception V3模型
        self.inception_model = None
        self._load_inception_model()
        
        # 自动准备real图片（避免重复准备）
        self._prepare_real_images_if_needed()

    def _load_inception_model(self):
        """加载预训练的Inception V3模型用于IS计算"""
        try:
            self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
            self.inception_model.eval()
            self.inception_model.to(self.device)
            print("Inception V3 model loaded for IS calculation")
        except Exception as e:
            print(f"Warning: Failed to load Inception V3 model: {e}")
            print("IS calculation will be disabled")

    def _prepare_real_images_if_needed(self, num_images=10000):
        """如果需要，准备real图片（避免重复生成）"""
        if FID_and_IS._shared_real_prepared:
            print(f"Shared real images already prepared in {self.real_dir}")
            return
        
        need_generate = True
        if os.path.exists(self.real_dir):
            real_files = [f for f in os.listdir(self.real_dir) if f.endswith('.png')]
            if len(real_files) >= num_images:
                need_generate = False
                print(f"Found {len(real_files)} real images in {self.real_dir}, skip generation.")
        
        if need_generate:
            print(f"Generating {num_images} real images to {self.real_dir} ...")
            self._generate_real_images(num_images)
            FID_and_IS._shared_real_prepared = True

    def _generate_real_images(self, num_images=10000):
        """生成并保存CIFAR-10测试集图片到统一目录"""
        # 清空目录
        if os.path.exists(self.real_dir):
            shutil.rmtree(self.real_dir)
        os.makedirs(self.real_dir, exist_ok=True)
        
        print(f"Preparing {num_images} real images...")
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        # 尝试不同的数据路径
        data_paths = ["../data/CIFAR10", "./data/CIFAR10", "~/data/CIFAR10"]
        dataset = None
        
        for data_path in data_paths:
            try:
                dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
                break
            except Exception as e:
                print(f"Failed to load CIFAR10 from {data_path}: {e}")
                continue
        
        if dataset is None:
            raise RuntimeError("Failed to load CIFAR10 dataset from any path")
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.real_batch_size, shuffle=False)
        count = 0
        for imgs, _ in loader:
            n = min(num_images - count, imgs.size(0))
            self._save_images(imgs[:n], self.real_dir, prefix="real")
            count += n
            if count >= num_images:
                break
        
        print(f"Successfully prepared {count} real images in {self.real_dir}")

    def _save_images(self, imgs, directory, prefix="img"):
        """Save a batch of images to a directory."""
        os.makedirs(directory, exist_ok=True)
        for i, img in enumerate(imgs):
            save_image(img, os.path.join(directory, f"{prefix}_{i:05d}.png"))

    def _save_images_with_offset(self, imgs, directory, prefix="img", start_idx=0):
        """保存图片，支持起始索引偏移"""
        os.makedirs(directory, exist_ok=True)
        
        for i, img in enumerate(imgs):
            img_path = os.path.join(directory, f"{prefix}_{start_idx + i:05d}.png")
            save_image(img, img_path)

    def prepare_fake_images(self, sampler, num_images=10000, batch_size=100, img_size=32, device=None):
        """
        使用采样器生成fake图片
        如果目录下已有足够的fake图片，则不再生成，避免重复生成。
        """
        # 检查现有fake图片数量
        existing_count = 0
        if os.path.exists(self.fake_dir):
            existing_files = [f for f in os.listdir(self.fake_dir) if f.endswith('.png')]
            existing_count = len(existing_files)
        else:
            os.makedirs(self.fake_dir, exist_ok=True)

        if existing_count >= num_images:
            print(f"Found {existing_count} fake images in {self.fake_dir}, skip generation.")
            return

        # 若不足，删除旧的，重新生成
        if existing_count > 0:
            print(f"Existing fake images ({existing_count}) less than required ({num_images}), regenerating all.")
            shutil.rmtree(self.fake_dir)
            os.makedirs(self.fake_dir, exist_ok=True)

        if device is None:
            device = self.device

        print(f"Generating {num_images} fake images for model '{self.model_name}' to {self.fake_dir}...")
        if self.con_model:
            print("Using conditional model with random labels (1-10)")
        else:
            print("Using unconditional model")

        sampler.eval()
        count = 0

        with torch.no_grad():
            while count < num_images:
                n = min(batch_size, num_images - count)
                noise = torch.randn(n, 3, img_size, img_size, device=device)

                # 生成图片
                if self.con_model:
                    # 条件模型：随机生成1-10的标签
                    labels = torch.randint(1, 11, (n,), device=device)
                    fake_imgs = sampler(noise, labels)
                else:
                    # 无条件模型
                    fake_imgs = sampler(noise)

                # 归一化到[0,1]
                if fake_imgs.min() < 0 or fake_imgs.max() > 1:
                    fake_imgs = (fake_imgs + 1) / 2

                # 保存图片
                start_idx = count
                self._save_images_with_offset(
                    fake_imgs.cpu(), self.fake_dir, prefix="fake", start_idx=start_idx
                )
                count += n

                if count % 1000 == 0:
                    print(f"Generated {count}/{num_images} fake images")

        print(f"Successfully generated {count} fake images")

    def compute_fid(self):
        """Compute FID score between saved real and fake images."""
        if not os.path.exists(self.real_dir):
            raise RuntimeError(f"Real images directory {self.real_dir} does not exist")
        
        if not os.path.exists(self.fake_dir):
            raise RuntimeError(f"Fake images directory {self.fake_dir} does not exist")
        
        real_files = [f for f in os.listdir(self.real_dir) if f.endswith('.png')]
        fake_files = [f for f in os.listdir(self.fake_dir) if f.endswith('.png')]
        
        print(f"Computing FID with {len(real_files)} real images and {len(fake_files)} fake images")
        print(f"Real dir: {self.real_dir}")
        print(f"Fake dir: {self.fake_dir}")
        
        try:
            fid_value = fid_score.calculate_fid_given_paths(
                [self.real_dir, self.fake_dir],
                batch_size=self.real_batch_size,
                device=self.device,
                dims=2048
            )
            return fid_value
        except Exception as e:
            raise RuntimeError(f"Failed to compute FID: {e}")

    def _load_images_from_directory(self, directory):
        """从目录加载图片用于IS计算"""
        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception V3输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        images = []
        image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
        image_files.sort()
        
        print(f"Loading {len(image_files)} images for IS calculation...")
        
        for img_file in image_files:
            img_path = os.path.join(directory, img_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
        
        return torch.stack(images)

    def _get_inception_predictions(self, images):
        """
        使用Inception V3获取预测概率
        
        Parameters:
        -----------
        images : torch.Tensor
            图片张量，形状为 (N, 3, H, W)
            
        Returns:
        --------
        predictions : np.ndarray
            预测概率，形状为 (N, 1000)
        """
        if self.inception_model is None:
            raise RuntimeError("Inception V3 model not loaded")
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(images), self.real_batch_size):
                batch = images[i:i + self.real_batch_size].to(self.device)
                
                # 获取预测
                pred = self.inception_model(batch)
                pred = F.softmax(pred, dim=1)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)

    def _calculate_is_from_predictions(self, predictions):
        """
        从预测概率计算Inception Score
        
        Parameters:
        -----------
        predictions : np.ndarray
            预测概率，形状为 (N, 1000)
            
        Returns:
        --------
        is_mean : float
            IS分数的均值
        is_std : float
            IS分数的标准差
        """
        # 分割数据计算标准差
        split_scores = []
        
        for k in range(self.is_splits):
            part = predictions[k * (len(predictions) // self.is_splits): 
                             (k + 1) * (len(predictions) // self.is_splits)]
            
            # 计算边际概率 p(y)
            py = np.mean(part, axis=0)
            
            # 计算KL散度并求和
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            
            split_scores.append(np.exp(np.mean(scores)))
        
        return np.mean(split_scores), np.std(split_scores)

    def compute_is(self):
        """
        计算保存的fake图片的IS分数
        
        Returns:
        --------
        is_mean : float
            IS分数的均值
        is_std : float
            IS分数的标准差
        """
        if self.inception_model is None:
            raise RuntimeError("Inception V3 model not loaded, cannot compute IS")
        
        if not os.path.exists(self.fake_dir):
            raise RuntimeError(f"Fake images directory {self.fake_dir} does not exist")
        
        # 加载图片
        images = self._load_images_from_directory(self.fake_dir)
        print(f"Computing IS for {len(images)} images")
        
        # 获取预测
        predictions = self._get_inception_predictions(images)
        
        # 计算IS
        is_mean, is_std = self._calculate_is_from_predictions(predictions)
        
        return is_mean, is_std

    def compute_is_from_tensors(self, images):
        """
        从张量直接计算IS
        
        Parameters:
        -----------
        images : torch.Tensor
            图片张量，形状为 (N, 3, H, W)，值域为 [0, 1]
            
        Returns:
        --------
        is_mean : float
            IS分数的均值
        is_std : float
            IS分数的标准差
        """
        if self.inception_model is None:
            raise RuntimeError("Inception V3 model not loaded, cannot compute IS")
        
        # 预处理图片
        if images.size(-1) != 299:  # Resize to 299x299 for Inception V3
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        
        # 获取预测
        predictions = self._get_inception_predictions(images)
        
        # 计算IS
        return self._calculate_is_from_predictions(predictions)

    def compute_both(self):
        """
        同时计算FID和IS分数
        
        Returns:
        --------
        results : dict
            包含FID和IS结果的字典
        """
        results = {}
        
        # 计算FID
        try:
            fid_value = self.compute_fid()
            results['fid'] = fid_value
            print(f"FID Score: {fid_value:.4f}")
        except Exception as e:
            print(f"FID calculation failed: {e}")
            results['fid'] = None
        
        # 计算IS
        try:
            is_mean, is_std = self.compute_is()
            results['is_mean'] = is_mean
            results['is_std'] = is_std
            print(f"IS Score: {is_mean:.4f} ± {is_std:.4f}")
        except Exception as e:
            print(f"IS calculation failed: {e}")
            results['is_mean'] = None
            results['is_std'] = None
        
        return results

    def cleanup_fake_images(self):
        """清理当前模型的fake图片"""
        if os.path.exists(self.fake_dir):
            shutil.rmtree(self.fake_dir)
            print(f"Cleaned up fake images for model '{self.model_name}' in {self.fake_dir}")

    def cleanup_all_fake(self):
        """清理整个tmp_dir下的所有fake图片"""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
            print(f"Cleaned up all fake images in {self.tmp_dir}")

    @classmethod
    def cleanup_shared_real(cls):
        """清理共享的real图片"""
        if os.path.exists(cls._shared_real_dir):
            shutil.rmtree(cls._shared_real_dir)
            cls._shared_real_prepared = False
            print(f"Cleaned up shared real images in {cls._shared_real_dir}")

    def get_stats(self):
        """获取统计信息"""
        real_count = len([f for f in os.listdir(self.real_dir) 
                         if f.endswith('.png')]) if os.path.exists(self.real_dir) else 0
        fake_count = len([f for f in os.listdir(self.fake_dir) 
                         if f.endswith('.png')]) if os.path.exists(self.fake_dir) else 0
        
        return {
            "model_name": self.model_name,
            "real_images": real_count,
            "fake_images": fake_count,
            "real_dir": self.real_dir,
            "fake_dir": self.fake_dir,
            "batch_size": self.real_batch_size,
            "is_splits": self.is_splits,
            "inception_loaded": self.inception_model is not None
        }


if __name__ == "__main__":
    pass
    # 示例用法
    # print("Testing FID_and_IS with unified directory structure...")
    
    # # 示例1: DDPM模型
    # ddpm_calculator = FID_and_IS(device="cuda", model_name="ddpm", tmp_dir="./tmp_fid_is")
    # print(f"DDPM Stats: {ddpm_calculator.get_stats()}")
    
    # # 示例2: Classifier Guidance模型
    # cg_calculator = FID_and_IS(device="cuda", model_name="classifier_guidance", tmp_dir="./tmp_fid_is")
    # print(f"CG Stats: {cg_calculator.get_stats()}")
    
    # # 示例3: 默认模型（fake图片直接放在tmp_dir下）
    # default_calculator = FID_and_IS(device="cuda", tmp_dir="./tmp_default")
    # print(f"Default Stats: {default_calculator.get_stats()}")
    
    # print("FID_and_IS Calculator with unified structure initialized successfully!")





# import os
# import shutil
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torchvision.utils import save_image
# from torchvision import datasets, transforms, models
# from pytorch_fid import fid_score
# from scipy.stats import entropy
# from PIL import Image
# import torch.multiprocessing as mp


# class FID_and_IS:
#     """
#     FID和IS计算器整合类
#     统一将real图片放在../data/realimg下，fake图片放在tmp_dir下
#     分布式计算：同时使用cuda:0和cuda:1两张GPU来解决显存不足问题
#     """
    
#     # 类级别的共享real图片目录
#     _shared_real_dir = "../data/realimg"
#     _shared_real_prepared = False
    
#     def __init__(self, devices=["cuda:0", "cuda:1"], model_name=None, 
#                  real_batch_size=50, tmp_dir="./tmp_fid_is", is_splits=10, con_model=False):
#         """
#         初始化FID和IS计算器
        
#         Parameters:
#         -----------
#         devices : list
#             用于计算的设备列表，例如 ["cuda:0", "cuda:1"]
#         model_name : str, optional
#             模型名称，用于区分不同模型的fake图片
#         real_batch_size : int
#             每个GPU上处理图片的批次大小（总批次大小将是 batch_size * len(devices)）
#         tmp_dir : str
#             临时文件目录（存放fake图片）
#         is_splits : int
#             计算IS时的分割数量，用于计算标准差
#         con_model : bool
#             是否为条件模型，如果为True则生成随机标签
#         """
#         self.devices = devices
#         self.num_devices = len(devices)
#         self.model_name = model_name or "default"
#         self.real_batch_size = real_batch_size  # 每个设备的批量大小
#         self.tmp_dir = tmp_dir
#         self.is_splits = is_splits
#         self.con_model = con_model
        
#         # 打印设备信息
#         print(f"Using {self.num_devices} devices for distributed computation: {self.devices}")
        
#         # 统一的real图片目录
#         self.real_dir = FID_and_IS._shared_real_dir
        
#         # fake图片直接放在tmp_dir下，按模型名区分
#         if self.model_name == "default":
#             self.fake_dir = self.tmp_dir
#         else:
#             self.fake_dir = os.path.join(self.tmp_dir, self.model_name)
        
#         # 创建必要的目录
#         os.makedirs(self.real_dir, exist_ok=True)
#         os.makedirs(self.fake_dir, exist_ok=True)
        
#         # 为IS计算加载Inception V3模型（在多个设备上）
#         self.inception_models = {}
#         self._load_inception_models()
        
#         # 自动准备real图片（避免重复准备）
#         self._prepare_real_images_if_needed()

#     def _load_inception_models(self):
#         """在每个设备上加载预训练的Inception V3模型用于IS计算"""
#         for device in self.devices:
#             try:
#                 model = models.inception_v3(pretrained=True, transform_input=False)
#                 model.eval()
#                 model.to(device)
#                 self.inception_models[device] = model
#                 print(f"Inception V3 model loaded for IS calculation on {device}")
#             except Exception as e:
#                 print(f"Warning: Failed to load Inception V3 model on {device}: {e}")
#                 print(f"IS calculation on {device} will be disabled")

#     def _prepare_real_images_if_needed(self, num_images=10000):
#         """如果需要，准备real图片（避免重复生成）"""
#         if FID_and_IS._shared_real_prepared:
#             print(f"Shared real images already prepared in {self.real_dir}")
#             return
        
#         need_generate = True
#         if os.path.exists(self.real_dir):
#             real_files = [f for f in os.listdir(self.real_dir) if f.endswith('.png')]
#             if len(real_files) >= num_images:
#                 need_generate = False
#                 print(f"Found {len(real_files)} real images in {self.real_dir}, skip generation.")
        
#         if need_generate:
#             print(f"Generating {num_images} real images to {self.real_dir} ...")
#             self._generate_real_images(num_images)
#             FID_and_IS._shared_real_prepared = True

#     def _generate_real_images(self, num_images=10000):
#         """生成并保存CIFAR-10测试集图片到统一目录"""
#         # 清空目录
#         if os.path.exists(self.real_dir):
#             shutil.rmtree(self.real_dir)
#         os.makedirs(self.real_dir, exist_ok=True)
        
#         print(f"Preparing {num_images} real images...")
        
#         transform = transforms.Compose([transforms.ToTensor()])
        
#         # 尝试不同的数据路径
#         data_paths = ["../data/CIFAR10", "./data/CIFAR10", "~/data/CIFAR10"]
#         dataset = None
        
#         for data_path in data_paths:
#             try:
#                 dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
#                 break
#             except Exception as e:
#                 print(f"Failed to load CIFAR10 from {data_path}: {e}")
#                 continue
        
#         if dataset is None:
#             raise RuntimeError("Failed to load CIFAR10 dataset from any path")
        
#         # 计算每个设备上处理的图片数量
#         total_batch_size = self.real_batch_size * self.num_devices
#         loader = torch.utils.data.DataLoader(
#             dataset, batch_size=total_batch_size, shuffle=False
#         )
        
#         count = 0
#         for imgs, _ in loader:
#             n = min(num_images - count, imgs.size(0))
#             self._save_images(imgs[:n], self.real_dir, prefix="real")
#             count += n
#             if count >= num_images:
#                 break
        
#         print(f"Successfully prepared {count} real images in {self.real_dir}")

#     def _save_images(self, imgs, directory, prefix="img"):
#         """Save a batch of images to a directory."""
#         os.makedirs(directory, exist_ok=True)
#         for i, img in enumerate(imgs):
#             save_image(img, os.path.join(directory, f"{prefix}_{i:05d}.png"))

#     def _save_images_with_offset(self, imgs, directory, prefix="img", start_idx=0):
#         """保存图片，支持起始索引偏移"""
#         os.makedirs(directory, exist_ok=True)
        
#         for i, img in enumerate(imgs):
#             img_path = os.path.join(directory, f"{prefix}_{start_idx + i:05d}.png")
#             save_image(img, img_path)

#     def prepare_fake_images(self, sampler, num_images=10000, batch_size=100, img_size=32, device=None):
#         """
#         使用采样器生成fake图片，在多GPU环境中自动分配生成任务
#         """
#         # 检查现有fake图片数量
#         existing_count = 0
#         if os.path.exists(self.fake_dir):
#             existing_files = [f for f in os.listdir(self.fake_dir) if f.endswith('.png')]
#             existing_count = len(existing_files)
#         else:
#             os.makedirs(self.fake_dir, exist_ok=True)

#         if existing_count >= num_images:
#             print(f"Found {existing_count} fake images in {self.fake_dir}, skip generation.")
#             return

#         # 若不足，删除旧的，重新生成
#         if existing_count > 0:
#             print(f"Existing fake images ({existing_count}) less than required ({num_images}), regenerating all.")
#             shutil.rmtree(self.fake_dir)
#             os.makedirs(self.fake_dir, exist_ok=True)

#         # 如果未指定设备，使用第一个设备
#         if device is None:
#             device = self.devices[0]

#         print(f"Generating {num_images} fake images for model '{self.model_name}' to {self.fake_dir}...")
#         if self.con_model:
#             print("Using conditional model with random labels (1-10)")
#         else:
#             print("Using unconditional model")

#         # 复制采样器到指定设备
#         sampler_device = sampler.to(device)
#         sampler_device.eval()
        
#         count = 0

#         with torch.no_grad():
#             while count < num_images:
#                 n = min(batch_size, num_images - count)
#                 noise = torch.randn(n, 3, img_size, img_size, device=device)

#                 # 生成图片
#                 if self.con_model:
#                     # 条件模型：随机生成1-10的标签
#                     labels = torch.randint(1, 11, (n,), device=device)
#                     fake_imgs = sampler_device(noise, labels)
#                 else:
#                     # 无条件模型
#                     fake_imgs = sampler_device(noise)

#                 # 归一化到[0,1]
#                 if fake_imgs.min() < 0 or fake_imgs.max() > 1:
#                     fake_imgs = (fake_imgs + 1) / 2

#                 # 保存图片
#                 start_idx = count
#                 self._save_images_with_offset(
#                     fake_imgs.cpu(), self.fake_dir, prefix="fake", start_idx=start_idx
#                 )
#                 count += n

#                 if count % 1000 == 0:
#                     print(f"Generated {count}/{num_images} fake images")

#         print(f"Successfully generated {count} fake images")

#     def compute_fid(self):
#         """
#         使用两个GPU计算FID分数，将计算负载分布在两个GPU上
#         我们通过修改内部实现来支持多GPU计算
#         """
#         if not os.path.exists(self.real_dir):
#             raise RuntimeError(f"Real images directory {self.real_dir} does not exist")
        
#         if not os.path.exists(self.fake_dir):
#             raise RuntimeError(f"Fake images directory {self.fake_dir} does not exist")
        
#         real_files = [f for f in os.listdir(self.real_dir) if f.endswith('.png')]
#         fake_files = [f for f in os.listdir(self.fake_dir) if f.endswith('.png')]
        
#         print(f"Computing FID with {len(real_files)} real images and {len(fake_files)} fake images")
#         print(f"Real dir: {self.real_dir}")
#         print(f"Fake dir: {self.fake_dir}")
#         print(f"Using multiple devices: {self.devices}")
        
#         try:
#             # 调用我们自定义的多GPU FID计算函数
#             fid_value = self.calculate_fid_given_paths_multi_gpu(
#                 [self.real_dir, self.fake_dir],
#                 batch_size=self.real_batch_size,  # 每个GPU的批量大小
#                 dims=2048
#             )
#             return fid_value
#         except Exception as e:
#             raise RuntimeError(f"Failed to compute FID: {e}")

#     def calculate_fid_given_paths_multi_gpu(self, paths, batch_size=50, dims=2048):
#         """
#         使用多GPU计算FID，将数据集分割到多个GPU上并行处理
#         这是对pytorch_fid.fid_score.calculate_fid_given_paths的修改版本
#         """
#         from pytorch_fid.inception import InceptionV3
        
#         print(f"Loading images for FID calculation, distributing across {len(self.devices)} devices")
        
#         # 初始化每个设备上的Inception V3模型
#         block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
#         inception_models = {}
#         for device in self.devices:
#             model = InceptionV3([block_idx]).to(device)
#             model.eval()
#             inception_models[device] = model
        
#         # 为每个路径计算激活
#         activations = []
#         for i, path in enumerate(paths):
#             print(f"\nCalculating activations for path {i+1}/{len(paths)}: {path}")
            
#             # 并行计算每个路径的激活
#             acts = self._get_activations_multi_gpu(
#                 path, inception_models, batch_size=batch_size
#             )
#             activations.append(acts)
        
#         # 计算FID
#         print("Calculating FID...")
#         act1, act2 = activations[0], activations[1]
#         mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
#         mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        
#         # 将统计数据移动到第一个GPU进行最终计算
#         device = self.devices[0]
#         fid_value = self._calculate_frechet_distance(
#             torch.tensor(mu1).to(device),
#             torch.tensor(sigma1).to(device),
#             torch.tensor(mu2).to(device),
#             torch.tensor(sigma2).to(device)
#         )
        
#         return fid_value

#     def _get_activations_multi_gpu(self, path, inception_models, batch_size=50):
#         """
#         分布在多个GPU上计算路径中所有图像的Inception激活
#         """
#         from torchvision.io import read_image
#         from pathlib import Path
        
#         # 获取所有图片文件
#         path = Path(path)
#         files = sorted([file for ext in ['png', 'jpg', 'jpeg'] 
#                         for file in path.glob(f'*.{ext}')])
        
#         # 创建预处理变换
#         preprocess = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize(299),
#             transforms.CenterCrop(299),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
        
#         # 计算每个设备处理的文件数
#         total_files = len(files)
#         files_per_device = total_files // len(self.devices)
        
#         # 每个设备处理一部分数据
#         all_preds = []
        
#         # 遍历所有文件，使用不同的设备处理不同的批次
#         for i in range(0, total_files, batch_size * len(self.devices)):
#             device_batches = []
            
#             # 为每个设备准备一个批次
#             for dev_idx, device in enumerate(self.devices):
#                 start_idx = i + dev_idx * batch_size
#                 end_idx = min(start_idx + batch_size, total_files)
                
#                 if start_idx >= total_files:
#                     break
                
#                 # 加载当前设备的一个批次
#                 batch_files = files[start_idx:end_idx]
#                 if not batch_files:
#                     continue
                    
#                 # 读取和预处理图像
#                 batch = []
#                 for file in batch_files:
#                     try:
#                         img = read_image(str(file))
#                         img = preprocess(img)
#                         batch.append(img)
#                     except Exception as e:
#                         print(f"Error processing {file}: {e}")
                
#                 if batch:
#                     # 创建批次张量并移动到相应设备
#                     batch_tensor = torch.stack(batch).to(device)
#                     device_batches.append((device, batch_tensor))
            
#             # 对每个设备上的批次并行计算预测
#             for device, batch_tensor in device_batches:
#                 with torch.no_grad():
#                     pred = inception_models[device](batch_tensor)[0]
#                     # 移动预测结果到CPU
#                     all_preds.append(pred.cpu().numpy().reshape(pred.shape[0], -1))
            
#             # 报告进度
#             if i % (10 * batch_size * len(self.devices)) == 0:
#                 print(f'Processed {min(i + batch_size * len(self.devices), total_files)}/{total_files} images')
        
#         # 合并所有预测
#         if not all_preds:
#             raise RuntimeError("No predictions were generated!")
            
#         return np.concatenate(all_preds, axis=0)

#     def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
#         """
#         计算Frechet距离
#         基于PyTorch的实现，计算发生在GPU上
#         """
#         diff = mu1 - mu2
        
#         # 产品可能不是对称的，所以强制转换
#         covmean, _ = self._sqrt_matrix_pa(sigma1 @ sigma2, eps=eps)
        
#         # 计算Frechet距离
#         tr_covmean = torch.trace(covmean)
        
#         return (torch.dot(diff, diff) + torch.trace(sigma1) +
#                 torch.trace(sigma2) - 2 * tr_covmean).cpu().item()

#     def _sqrt_matrix_pa(self, A, eps=1e-6):
#         """
#         计算方阵的平方根
#         使用SVD分解的GPU实现
#         """
#         device = A.device
#         U, s, Vh = torch.linalg.svd(A)
        
#         # 防止数值不稳定性
#         mask = s > eps
#         s_sqrt = torch.zeros_like(s)
#         s_sqrt[mask] = torch.sqrt(s[mask])
        
#         return U @ torch.diag(s_sqrt) @ Vh, U, s_sqrt, Vh

#     def _load_images_from_directory(self, directory):
#         """从目录加载图片用于IS计算"""
#         transform = transforms.Compose([
#             transforms.Resize((299, 299)),  # Inception V3输入尺寸
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                std=[0.229, 0.224, 0.225])
#         ])
        
#         images = []
#         image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
#         image_files.sort()
        
#         print(f"Loading {len(image_files)} images for IS calculation...")
        
#         for img_file in image_files:
#             img_path = os.path.join(directory, img_file)
#             img = Image.open(img_path).convert('RGB')
#             img_tensor = transform(img)
#             images.append(img_tensor)
        
#         return torch.stack(images)

#     def _get_inception_predictions_multi_gpu(self, images):
#         """
#         使用多GPU获取预测概率，将图像分布在不同GPU上进行计算
        
#         Parameters:
#         -----------
#         images : torch.Tensor
#             图片张量，形状为 (N, 3, H, W)
            
#         Returns:
#         --------
#         predictions : np.ndarray
#             预测概率，形状为 (N, 1000)
#         """
#         if not self.inception_models:
#             raise RuntimeError("Inception V3 models not loaded on any device")
        
#         predictions = []
        
#         # 计算每个设备处理的批次大小
#         total_images = len(images)
#         per_device_batch = self.real_batch_size
        
#         with torch.no_grad():
#             # 按批次处理图像，轮流使用不同设备
#             for i in range(0, total_images, per_device_batch * self.num_devices):
#                 # 对每个设备分配一个批次
#                 for dev_idx, device in enumerate(self.devices):
#                     start_idx = i + dev_idx * per_device_batch
#                     end_idx = min(start_idx + per_device_batch, total_images)
                    
#                     if start_idx >= total_images:
#                         break
                        
#                     # 获取当前批次并移动到对应设备
#                     batch = images[start_idx:end_idx].to(device)
                    
#                     # 获取预测
#                     inception_model = self.inception_models[device]
#                     pred = inception_model(batch)
#                     pred = F.softmax(pred, dim=1)
                    
#                     # 将预测结果移回CPU并添加到列表中
#                     predictions.append(pred.cpu().numpy())
                    
#                 # 显示进度
#                 if i % (10 * per_device_batch * self.num_devices) == 0:
#                     print(f'IS calculation: Processed {min(i + per_device_batch * self.num_devices, total_images)}/{total_images} images')
        
#         return np.concatenate(predictions, axis=0)

#     def _calculate_is_from_predictions(self, predictions):
#         """
#         从预测概率计算Inception Score
        
#         Parameters:
#         -----------
#         predictions : np.ndarray
#             预测概率，形状为 (N, 1000)
            
#         Returns:
#         --------
#         is_mean : float
#             IS分数的均值
#         is_std : float
#             IS分数的标准差
#         """
#         # 分割数据计算标准差
#         split_scores = []
        
#         for k in range(self.is_splits):
#             part = predictions[k * (len(predictions) // self.is_splits): 
#                              (k + 1) * (len(predictions) // self.is_splits)]
            
#             # 计算边际概率 p(y)
#             py = np.mean(part, axis=0)
            
#             # 计算KL散度并求和
#             scores = []
#             for i in range(part.shape[0]):
#                 pyx = part[i, :]
#                 scores.append(entropy(pyx, py))
            
#             split_scores.append(np.exp(np.mean(scores)))
        
#         return np.mean(split_scores), np.std(split_scores)

#     def compute_is(self):
#         """
#         使用多GPU计算保存的fake图片的IS分数
        
#         Returns:
#         --------
#         is_mean : float
#             IS分数的均值
#         is_std : float
#             IS分数的标准差
#         """
#         if not self.inception_models:
#             raise RuntimeError("Inception V3 models not loaded, cannot compute IS")
        
#         if not os.path.exists(self.fake_dir):
#             raise RuntimeError(f"Fake images directory {self.fake_dir} does not exist")
        
#         # 加载图片
#         images = self._load_images_from_directory(self.fake_dir)
#         print(f"Computing IS for {len(images)} images using {self.num_devices} devices")
        
#         # 使用多GPU获取预测
#         predictions = self._get_inception_predictions_multi_gpu(images)
        
#         # 计算IS
#         is_mean, is_std = self._calculate_is_from_predictions(predictions)
        
#         return is_mean, is_std

#     def compute_is_from_tensors(self, images):
#         """
#         从张量直接计算IS，使用多GPU
        
#         Parameters:
#         -----------
#         images : torch.Tensor
#             图片张量，形状为 (N, 3, H, W)，值域为 [0, 1]
            
#         Returns:
#         --------
#         is_mean : float
#             IS分数的均值
#         is_std : float
#             IS分数的标准差
#         """
#         if not self.inception_models:
#             raise RuntimeError("Inception V3 models not loaded, cannot compute IS")
        
#         # 预处理图片
#         if images.size(-1) != 299:  # Resize to 299x299 for Inception V3
#             images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
#         # 标准化
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
#         images = (images - mean) / std
        
#         # 使用多GPU进行预测
#         predictions = self._get_inception_predictions_multi_gpu(images)
        
#         # 计算IS
#         return self._calculate_is_from_predictions(predictions)

#     def compute_both(self):
#         """
#         同时计算FID和IS分数，两者都使用多GPU分布式计算
        
#         Returns:
#         --------
#         results : dict
#             包含FID和IS结果的字典
#         """
#         results = {}
        
#         # 计算FID (使用多GPU)
#         try:
#             print(f"Starting FID calculation using devices: {self.devices}")
#             fid_value = self.compute_fid()
#             results['fid'] = fid_value
#             print(f"FID Score: {fid_value:.4f}")
#         except Exception as e:
#             print(f"FID calculation failed: {e}")
#             results['fid'] = None
        
#         # 计算IS (使用多GPU)
#         try:
#             print(f"Starting IS calculation using devices: {self.devices}")
#             is_mean, is_std = self.compute_is()
#             results['is_mean'] = is_mean
#             results['is_std'] = is_std
#             print(f"IS Score: {is_mean:.4f} ± {is_std:.4f}")
#         except Exception as e:
#             print(f"IS calculation failed: {e}")
#             results['is_mean'] = None
#             results['is_std'] = None
        
#         return results

#     def cleanup_fake_images(self):
#         """清理当前模型的fake图片"""
#         if os.path.exists(self.fake_dir):
#             shutil.rmtree(self.fake_dir)
#             print(f"Cleaned up fake images for model '{self.model_name}' in {self.fake_dir}")

#     def cleanup_all_fake(self):
#         """清理整个tmp_dir下的所有fake图片"""
#         if os.path.exists(self.tmp_dir):
#             shutil.rmtree(self.tmp_dir)
#             print(f"Cleaned up all fake images in {self.tmp_dir}")

#     @classmethod
#     def cleanup_shared_real(cls):
#         """清理共享的real图片"""
#         if os.path.exists(cls._shared_real_dir):
#             shutil.rmtree(cls._shared_real_dir)
#             cls._shared_real_prepared = False
#             print(f"Cleaned up shared real images in {cls._shared_real_dir}")

#     def get_stats(self):
#         """获取统计信息"""
#         real_count = len([f for f in os.listdir(self.real_dir) 
#                          if f.endswith('.png')]) if os.path.exists(self.real_dir) else 0
#         fake_count = len([f for f in os.listdir(self.fake_dir) 
#                          if f.endswith('.png')]) if os.path.exists(self.fake_dir) else 0
        
#         return {
#             "model_name": self.model_name,
#             "real_images": real_count,
#             "fake_images": fake_count,
#             "real_dir": self.real_dir,
#             "fake_dir": self.fake_dir,
#             "batch_size_per_device": self.real_batch_size,
#             "total_batch_size": self.real_batch_size * self.num_devices,
#             "is_splits": self.is_splits,
#             "devices": self.devices,
#             "num_devices": self.num_devices,
#             "inception_models_loaded": [device for device in self.devices if device in self.inception_models]
#         }


# if __name__ == "__main__":
#     # 示例用法 - 使用两个GPU进行分布式计算
#     print("Testing FID_and_IS with distributed multi-GPU setup...")
    
#     # 指定两个GPU设备
#     devices = ["cuda:0", "cuda:1"]
    
#     # 示例1: DDPM模型 - 所有计算分布在两个GPU上
#     ddpm_calculator = FID_and_IS(devices=devices, model_name="ddpm", 
#                                 real_batch_size=50, tmp_dir="./tmp_fid_is")
#     print(f"DDPM Stats: {ddpm_calculator.get_stats()}")
    
#     # # 示例2: Classifier Guidance模型
#     # cg_calculator = FID_and_IS(devices=devices, model_name="classifier_guidance", 
#     #                           real_batch_size=50, tmp_dir="./tmp_fid_is")
#     # print(f"CG Stats: {cg_calculator.get_stats()}")
    
#     # # 示例3: 默认模型
#     # default_calculator = FID_and_IS(devices=devices, real_batch_size=50,
#     #                               tmp_dir="./tmp_default")
#     # print(f"Default Stats: {default_calculator.get_stats()}")
    
#     print("FID_and_IS Calculator with distributed multi-GPU setup initialized successfully!")