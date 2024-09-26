import torch
import models
import os
from experiment import Experiment
from data_loader import HotdogDataLoader
import yaml
from saliency_map import compute_smooth_grad, plot_saliency_map, compute_integrated_gradients

def load_config(config_path='assignment1/confog.yaml'):
    config_path = os.path.normpath(config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
  config = load_config()
  # # for debug
  # config = load_config('assignment1/confog_debug.yaml')

  print("Experiment Arguments:")
  print(yaml.dump(config, default_flow_style=False))

  model_config = config['model']
  data_config = config['data']
  optimizer_config = config['optimizer']
  training_config = config['training']

  learning_rate = float(optimizer_config['lr'])
  batch_size = int(training_config['batch_size'])

  if model_config['name'] == 'VGG16':
      model = models.VGG16(int(model_config['num_classes']))
  elif model_config['name'] == 'Resnet18':
      model = models.Resnet18
      model.fc = torch.nn.Linear(model.fc.in_features, 2)
  else:
      raise ValueError(f"Unsupported model: {model_config['name']}")

  model.to(device)
  print(model)

  # Select optimizer
  if optimizer_config['name'] == 'Adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  elif optimizer_config['name'] == 'SGD':
      optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  else:
      raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")

  exp = Experiment(model,
                   optimizer,
                   num_epochs=int(training_config['epochs']),
                   batch_size=batch_size)

  dl = HotdogDataLoader(int(data_config['image_size']),
                         bool(data_config['augment']),
                         batch_size,
                         float(data_config['validation_split']))
  exp.run(dl)
  train_loader,_,_,_,_,_ = dl.data_for_exp()
  img, label = next(iter(train_loader))
  img = img[0].unsqueeze(0)
  label = label[0].unsqueeze(0)
  saliency, noise_level = compute_smooth_grad(model, img.to(device), label.to(device), num_samples=50, sigma=0.1)
  integrated_gradients = compute_integrated_gradients(model, img.to(device), label.to(device))
  print('noise_level', noise_level)
  plot_saliency_map(img, saliency, integrated_gradients, exp.solve_path('results', 'saliency.png'))
