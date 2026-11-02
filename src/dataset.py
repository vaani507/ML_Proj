import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class SolarCellDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load RGB image
        image = Image.open(img_path).convert('RGB')
        
        # Load grayscale binary mask (0 and 1)
        mask = Image.open(mask_path).convert('L')
        mask_arr = np.array(mask, dtype=np.float32)

        # Defect classification label (1.0 if defect pixels > 0, else 0.0)
        cls_label = 1.0 if np.sum(mask_arr) > 0 else 0.0

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)  # Shape: (1, 224, 224)
        cls_tensor = torch.tensor(cls_label, dtype=torch.float32)

        return image, mask_tensor, cls_tensor

def get_dataloaders(root_processed_dir, batch_size=8):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_img_dir = os.path.join(root_processed_dir, "train", "images")
    train_mask_dir = os.path.join(root_processed_dir, "train", "masks")
    test_img_dir = os.path.join(root_processed_dir, "test", "images")
    test_mask_dir = os.path.join(root_processed_dir, "test", "masks")

    train_dataset = SolarCellDataset(train_img_dir, train_mask_dir, transform=transform)
    test_dataset = SolarCellDataset(test_img_dir, test_mask_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader