import os
import numpy as np
import glob
import PIL.Image as Image
from torch.utils.data import random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Hotdog_NotHotdog(torch.utils.data.Dataset):
  def __init__(self, train, transform):
    'Initialization'
    self.transform = transform
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'hotdog_nothotdog')
    data_path = os.path.join(data_dir, 'train' if train else 'test')
    image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
    image_classes.sort()
    self.name_to_label = {c: id for id, c in enumerate(image_classes)}
    self.image_paths = glob.glob(data_path + '/*/*.jpg')

  def __len__(self):
    'Returns the total number of samples'
    return len(self.image_paths)

  def __getitem__(self, idx):
    'Generates one sample of data'
    image_path = self.image_paths[idx]

    image = Image.open(image_path)
    c = os.path.split(os.path.split(image_path)[0])[1]
    y = self.name_to_label[c]
    X = self.transform(image)
    return X, y

class HotdogDataLoader:
  def __init__(self, img_size = 32, augment=False, batch_size=64, validation_split=0.2):
    self.batch_size = batch_size
    self.validation_split = validation_split
    if augment:
      self.train_transform = transforms.Compose([
                                    transforms.Resize((img_size + 8, img_size + 8)),
                                    transforms.RandomCrop((img_size, img_size)),
                                    transforms.RandomRotation(15, expand=False),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.CenterCrop(img_size),
                                    transforms.ToTensor()
                             ])
    else:
      self.train_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                      transforms.ToTensor()])

    self.test_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                      transforms.ToTensor()])

  def data_for_exp(self):
    """Gemerate data for experiment

    Returns:
        train_loader, val_loader, test_loader, trainset, valset, testset
    """
    batch_size = self.batch_size
    val_split = self.validation_split

    full_trainset = Hotdog_NotHotdog(train=True, transform=self.train_transform)

    val_size = int(len(full_trainset) * val_split)
    train_size = len(full_trainset) - val_size

    trainset, valset = random_split(full_trainset, [train_size, val_size])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=3)

    testset = Hotdog_NotHotdog(train=False, transform=self.test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

    return train_loader, val_loader, test_loader, trainset, valset, testset

if __name__ == "__main__":
  print(f"Data in progress")
  dl = HotdogDataLoader(32, True, 64, 0.2)
  train_loader, val_loader, test_loader, trainset, valset, testset = dl.data_for_exp()
  images, labels = next(iter(train_loader))

  plt.figure(figsize=(20,10))

  for i in range(21):
      plt.subplot(5,7,i+1)
      plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
      plt.title(['hotdog', 'not hotdog'][labels[i].item()])
      plt.axis('off')
  plt.show()

  dl2 = HotdogDataLoader(32, False, 64, 0.2)
  train_loader, _, _, _, _, _ = dl2.data_for_exp()
  mean = 0.0
  std = 0.0
  nb_samples = 0.0
  for data, _ in train_loader:
      batch_samples = data.size(0)
      data = data.view(batch_samples, data.size(1), -1)
      mean += data.mean(2).sum(0)
      std += data.std(2).sum(0)
      nb_samples += batch_samples

  mean /= nb_samples
  std /= nb_samples

  computed_mean = mean.tolist()
  computed_std = std.tolist()

  print(f'mean: {computed_mean}')
  print(f'std: {computed_std}')
