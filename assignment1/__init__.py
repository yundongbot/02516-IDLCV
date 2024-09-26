import torch
import models
from experiment import Experiment
from data_loader import Hotdog_DataLoader
import yaml

def load_config(config_path='assignment1/confog.yaml'):
    import os
    config_path = os.path.normpath(config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
  config = load_config()

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
  # for debug
  # exp = Experiment(model, optimizer, num_epochs=1)
  dl = Hotdog_DataLoader(int(data_config['image_size']),
                         bool(data_config['augment']),
                         batch_size,
                         float(data_config['validation_split']))
  exp.run(dl)
