#  Fish Species Identification (Bangladesh)

This project presents a deep learning solution for identifying **20 common fish species** found in local markets across Bangladesh. Using high-resolution smartphone images collected from these markets, we trained and evaluated CNN-based models such as **ResNet** for multi-class fish classification.

---

## 📁 Dataset Overview

The dataset contains a total of **50,000 images**, split into **26,950 raw images** and **23,050 augmented images** across 20 species. **This project uses only the raw images** to preserve the integrity of real-world visuals without synthetic alterations.

### 🔹 Source

- 📦 **Zip File**: `Fish Data.zip` (6.67 GB)
- 📂 **Main Folder after extraction**: `Fish Data/`
- 📄 Each subfolder = 1 fish species

> 💡 No preprocessing or normalization was applied to the raw dataset prior to training.

---

## 🐠 Fish Species and Raw Image Counts

| Species                       | Raw Images |
|------------------------------|------------|
| Aair (Long Whiskers Catfish) | 1,804      |
| Boal (Wallago)               | 1,651      |
| Chapila (Indian River Shad)  | 428        |
| DeshiPuti (Local Puti)       | 412        |
| Foli (Spotted Snakehead)     | 562        |
| Ilish (Hilsa)                | 1,031      |
| KalBaush (Orangefin Labeo)   | 917        |
| Katla (Catla)                | 1,765      |
| Koi (Climbing Perch)         | 842        |
| Magur (Walking Catfish)      | 574        |
| Mrigel (Mrigal)              | 1,808      |
| Pabda (Pabdah Catfish)       | 1,764      |
| Pangas (Pangas Catfish)      | 934        |
| Puti (Puti)                  | 1,560      |
| Rui (Rohu)                   | 2,500      |
| Shol (Snakehead Murrel)      | 1,424      |
| Taki (Spotted Snakehead)     | 2,223      |
| Tarabaim (Striped Snakehead) | 1,262      |
| Telapiya (Tilapia)           | 2,058      |
| Tengra (Tengra Catfish)      | 1,431      |

---

## 🧠 Models Used

- ✅ ResNet-50

All models were fine-tuned using **ImageNet pre-trained weights** and modified to output predictions for **20 classes**.

---

## 🛠️ Data Loading (PyTorch)

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root='Fish Data/', transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
