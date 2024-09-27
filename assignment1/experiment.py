import os
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
import random
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

class Experiment:
  def __init__(self, model, optimizer, dataloader, num_epochs=100, batch_size=64,):
    self.model = model
    self.optimizer = optimizer
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.dataloader = dataloader
    self.dataset = self.dataloader.data_for_exp()
    self.id = f'{self.model.name}{datetime.now().strftime("-%m%d%H%M%S")}{random.randint(0, 1000)}'

  def run(self):
    """Run an experiment and then save the results to a csv file and a plot

    Returns:
        dict: Results of the experiment
    """
    train_loader, val_loader, test_loader, trainset, valset, testset = self.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = self.model
    optimizer = self.optimizer
    loss_fun = nn.CrossEntropyLoss()
    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}

    # ============ Training loop ============
    for epoch in tqdm(range(self.num_epochs), unit='epoch'):
      model.train()
      #For each epoch
      train_correct = 0
      train_loss = []
      for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
          data, target = data.to(device), target.to(device)
          #Zero the gradients computed for each weight
          optimizer.zero_grad()
          #Forward pass your image through the network
          output = model(data)
          #Compute the loss
          loss = loss_fun(output, target)
          #Backward pass through the network
          loss.backward()
          #Update the weights
          optimizer.step()

          train_loss.append(loss.item())
          #Compute how many were correctly classified
          predicted = output.argmax(1)
          train_correct += (target==predicted).sum().cpu().item()

      # ============ Validation loop ============
      model.eval()

      test_loss = []
      test_correct = 0
      with torch.no_grad():
        for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          test_loss.append(loss_fun(output, target).cpu().item())
          predicted = output.argmax(1)
          test_correct += (target==predicted).sum().cpu().item()
      out_dict['train_acc'].append(train_correct/len(trainset))
      out_dict['test_acc'].append(test_correct/len(testset))
      out_dict['train_loss'].append(np.mean(train_loss))
      out_dict['test_loss'].append(np.mean(test_loss))
      print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
            f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
    self.to_csv(out_dict)
    self.to_pic(out_dict)
    self.save_model(model)

    return out_dict

  def solve_path(self, path, file_type):
    """Solve the path to save the results

    Args:
        path (str): Path to save the results
        file_type (str): Type of the file

    Returns:
        str: Path to save the results
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, path)
    os.makedirs(data_dir, exist_ok=True)
    file = f'{data_dir}/{self.id}.{file_type}'
    return file

  def save_model(self, model):
    file = self.solve_path('results', 'pth')
    torch.save(model.state_dict(), file)

  def to_pic(self, results):
    """Save the results to a plot

    Args:
        results (dict): Results of the experiment
        path (str): Path to save the results
    """
    epochs = range(1, len(results['train_acc']) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_acc'], label='Train Accuracy')
    plt.plot(epochs, results['test_acc'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_loss'], label='Train Loss')
    plt.plot(epochs, results['test_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.tight_layout()
    file = self.solve_path('results', 'png')
    plt.savefig(file)
    plt.close()

  def to_csv(self, results):
    """Save the results to a csv file

    Args:
        results (dict): Results of the experiment
    """
    df = pd.DataFrame(results)
    file = self.solve_path('results', 'csv')
    df.to_csv(file, index=False)
