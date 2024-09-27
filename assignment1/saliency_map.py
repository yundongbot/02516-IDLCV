from matplotlib import pyplot as plt
import torch
import numpy as np
from data_loader import HotdogDataLoader
import torch.multiprocessing as mp
from models import VGG16
import torch.nn as nn

def normalize(saliency: torch.Tensor):
    return (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

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
  return torch.stack(images, dim=0)

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
    loss = outputs[:, target].sum()
    model.zero_grad()
    loss.backward()

    smooth_grad = imgs.grad.data.mean(dim=0, keepdim=True)
    # No matter the sign, we take the absolute value
    # since we only care about the magnitude of the influence
    # of each pixel on the prediction, not its direction
    saliency = smooth_grad.abs().squeeze().detach().cpu().numpy()

    if saliency.shape[0] == 3:
        saliency = np.sum(saliency, axis=0)

    noise_level = sigma / (imgs.max() - imgs.min())

    return normalize(saliency), noise_level

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
    return torch.stack(images, dim=0)

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
    print(imgs.shape) # torch.Size([51, 3, 64, 64])
    print(target.shape) # torch.Size([])
    imgs.requires_grad = True
    model.eval()

    outputs = model(imgs)
    loss = outputs[:, target].sum()
    print('outputs',outputs.shape) # outputs torch.Size([51, 2])
    print('loss',loss.shape) # loss torch.Size([])
    model.zero_grad()
    loss.backward()

    integrated_grads = imgs.grad.data.mean(dim=0, keepdim=True)
    integrated_grads = (image - baseline) * integrated_grads
    integrated_grads = integrated_grads.abs().squeeze().detach().cpu().numpy()

    if integrated_grads.ndim == 3:
        integrated_grads = np.mean(integrated_grads, axis=0)

    # normalize saliency map
    return normalize(integrated_grads)

def plot_saliency_map(img, smooth_grad, integrated_gradients,path = None):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img.squeeze().permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(smooth_grad, cmap=plt.cm.gray)
    plt.title('Smooth Grad')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(integrated_gradients, cmap=plt.cm.gray)
    plt.title('Integrated Gradients')
    plt.axis('off')
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()

def get_last_conv_layer(model) -> tuple[nn.Module, int]:
    """Get the last convolutional layer of the model

    Args:
        model (torch.nn.Module): Model to be used

    Returns:
        torch.nn.Module: Last convolutional layer
        int: Index of the last convolutional layer
    """
    # iterate through all layers
    for i in range(len(model.net) - 1, -1, -1):
      if isinstance(model.net[i], nn.Conv2d):
        return model.net[i], i
    return None, None

def compute_grad_cam(model, image, target):
    """Compute the grad cam of the image

    Args:
        model (torch.nn.Module): Model to be used
        image (torch.Tensor): Image to be smoothed
        target (torch.Tensor): Target of the image
    """
    image = image.unsqueeze(0)
    last_conv_layer, _ = get_last_conv_layer(model)

    gradients = {}
    activations = {}

    def forward_hook(module, input, output):
        activations['value'] = output
        output.register_hook(lambda grad: gradients.update({'value': grad}))

    last_conv_layer.register_forward_hook(forward_hook)

    output = model(image)
    loss = output[0, target]
    model.zero_grad()
    loss.backward()

    grads = gradients['value']
    fmap = activations['value']

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)

    grad_cam = torch.sum(weights * fmap, dim=1)
    grad_cam = torch.relu(grad_cam)
    grad_cam = grad_cam.squeeze(0).detach().cpu().numpy()

    return normalize(grad_cam)

def plot_saliency_hot_map(img, grad_cam):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img.squeeze().permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(grad_cam, cmap=plt.cm.jet)
    plt.title('Smooth Grad')
    plt.axis('off')
    plt.show()

def main():
    model = VGG16(num_classes=2)
    model.load_state_dict(torch.load('./assignment1/results/model.pth', map_location=torch.device('cpu')))
    dl = HotdogDataLoader(64, False, 64, 0.2)
    train_loader, _, _, _, _, _ = dl.data_for_exp()
    images, labels = next(iter(train_loader))

    img = images[0]
    target = labels[0]
    print(labels.shape)
    print(target.shape)
    print(target)
    grad_cam = compute_grad_cam(model, img.clone().detach(), target)
    # saliency, noise_level = compute_smooth_grad(model, img.clone().detach(), target)
    # integrated_gradients = compute_integrated_gradients(model, img.clone().detach(), target)
    # print('noise_level', noise_level)
    # plot_saliency_map(img, saliency, integrated_gradients)
    plot_saliency_hot_map(img, grad_cam)

if __name__ == '__main__':
    mp.freeze_support()  # This is necessary for Windows compatibility
    main()
