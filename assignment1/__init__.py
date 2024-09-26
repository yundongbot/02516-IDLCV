import sys
import torch
import models
import argparse
from experiment import Experiment
from data_loader import Hotdog_DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dl = Hotdog_DataLoader(32)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Experiment configuration')
  parser.add_argument('--model', type=str, default='VGG16', help='Model name (e.g., VGG16, Resnet18)')
  parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer name (default: Adam)')
  parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
  parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')

  args = parser.parse_args()

  # Select model
  if args.model == 'VGG16':
      model = models.VGG16(2)
  elif args.model == 'Resnet18':
      model = models.Resnet18
      model.fc = torch.nn.Linear(model.fc.in_features, 2)
  else:
      raise ValueError(f"Unsupported model: {args.model}")

  model.to(device)
  print(model)

  # Select optimizer
  if args.optimizer == 'Adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  elif args.optimizer == 'SGD':
      optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
  else:
      raise ValueError(f"Unsupported optimizer: {args.optimizer}")

  print("Experiment Arguments:")
  for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
  print()  # Add an empty line for better readability

  exp = Experiment(model, optimizer, num_epochs=args.epochs, batch_size=args.batch_size)
  # for debug
  # exp = Experiment(model, optimizer, num_epochs=1)
  exp.run(dl)
