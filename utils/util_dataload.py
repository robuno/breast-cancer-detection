import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

NUM_WORKERS = os.cpu_count()

def create_dataloader(
    file_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    shuffle_opt: bool, 
    num_workers: int=NUM_WORKERS ):

  file_data = datasets.ImageFolder(file_dir, transform=transform)

  data_loader = DataLoader(
      file_data,
      batch_size=batch_size,
      shuffle=shuffle_opt,
      num_workers=num_workers,
      pin_memory=True, )

  return data_loader

def create_dataloader_w_weights(
    file_dir, 
    transform, 
    batch_size,  
    sampler,
    num_workers= NUM_WORKERS
    ):

  file_data = datasets.ImageFolder(file_dir, transform=transform)

  data_loader = DataLoader(
      file_data,
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True, 
      sampler= sampler)

  return data_loader

def get_class_names_dloader(file_dir: str, transform: transforms.Compose):
  file_data = datasets.ImageFolder(file_dir, transform=transform)
  class_names = file_data.classes

  return class_names