from matplotlib import pyplot as plt
import os
import torch
import numpy as np
from data_loader import HotdogDataLoader
import torch.multiprocessing as mp
from models import VGG16

def generate_smoothed_image(image, num_samples, sigma):
  """Generate smoothed images with noise

  Args:
      image (torch.Tensor): Image to be smoothed
      num_samples (int): Number of smoothed images to generate
      sigma (float): Standard deviation of the noise

  Returns:
      torch.Tensor: Smoothed images
  """
  images = []
  for _ in range(num_samples):
      noise = torch.randn_like(image) * sigma
      noisy_image = image + noise
      noisy_image = torch.clamp(noisy_image, 0, 1)
      images.append(noisy_image)
  return torch.cat(images, dim=0)

def compute_smooth_grad(model, image, target, num_samples=50, sigma=0.1):
    """Compute the smooth gradient of the image

    Args:
        model (torch.nn.Module): Model to be used
        image (torch.Tensor): Image to be smoothed
        target (torch.Tensor): Target of the image
        num_samples (int): Number of smoothed images to generate
        sigma (float): Standard deviation of the noise

    Returns:
        torch.Tensor: Smoothed images
    """
    imgs = generate_smoothed_image(image, num_samples, sigma)
    model.eval()
    # track differentiation
    imgs.requires_grad = True
    outputs = model(imgs)
    loss = outputs[0, target]

    model.zero_grad()
    loss.backward()

    smooth_grad = imgs.grad.data.mean(dim=0, keepdim=True)
    saliency = smooth_grad.abs().squeeze().detach().cpu().numpy()

    if saliency.shape[0] == 3:
        saliency = np.sum(saliency, axis=0)

    noise_level = sigma / (imgs.max() - imgs.min())

    # normalize saliency map
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    return saliency, noise_level

def generate_baseline_images(image, baseline, num_samples=50):
    """Generate baseline images

    Args:
        image (torch.Tensor): Image to be smoothed
        num_samples (int): Number of baseline images to generate
    """
    images = []
    factor = np.linspace(0, 1, num_samples+1)
    for a in factor:
      img = baseline * a * (image - baseline)
      images.append(img)
    return torch.cat(images, dim=0)

def compute_integrated_gradients(model, image, target, baseline = None, num_samples=50):
    """Compute the integrated gradients of the image

    Args:
        model (torch.nn.Module): Model to be used
        image (torch.Tensor): Image to be smoothed
        target (torch.Tensor): Target of the image
        num_samples (int): Number of smoothed images to generate
    """
    if baseline is None:
        baseline = torch.zeros_like(image)
    imgs = generate_baseline_images(image, baseline, num_samples)
    model.eval()
    imgs.requires_grad = True

    outputs = model(imgs)
    loss = outputs[0, target]
    model.zero_grad()
    loss.backward()

    avg_grads = imgs.grad.data.mean(dim=0, keepdim=True)
    integrated_grads = (image - baseline) * avg_grads
    integrated_grads = integrated_grads.squeeze().detach().cpu().numpy()

    if integrated_grads.ndim == 3:
        integrated_grads = np.mean(integrated_grads, axis=0)

    # normalize saliency map
    integrated_grads = (integrated_grads - integrated_grads.min()) / (integrated_grads.max() - integrated_grads.min())

    return integrated_grads

def plot_saliency_map(img, saliency, path = None):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img.squeeze().permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(saliency, cmap=plt.cm.gray)
    plt.title('Saliency Map')
    plt.axis('off')
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()

def main():
    model = VGG16(num_classes=2)
    model.load_state_dict(torch.load('./assignment1/results/VGG16-2024-09-26_23-02-17-936.pth', map_location=torch.device('cpu')))
    dl = HotdogDataLoader(64, False, 64, 0.2)
    train_loader, _, _, _, _, _ = dl.data_for_exp()
    images, labels = next(iter(train_loader))

    img = images[0].unsqueeze(0)
    target = labels[0].unsqueeze(0)

    saliency = compute_integrated_gradients(model, img, target)
    # print('noise_level', noise_level)
    print(saliency)
    plot_saliency_map(img, saliency)

if __name__ == '__main__':
    mp.freeze_support()  # This is necessary for Windows compatibility
    main()
