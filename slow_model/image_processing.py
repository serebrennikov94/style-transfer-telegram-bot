import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL.Image import Image


def image_loader(image: Image,
                 imsize: int,
                 device: torch.device) -> torch.Tensor:
    """
    Get the image and transform it to torch.Tensor

    Parameters
    ----------
    image (Image):
        Image from PIL library
    imsize (int):
        Size of image
    device (torch.device):
        Pytorch device ('cuda' or 'cpu') for calculations
    Returns
    -------
        torch.Tensor with image
    """
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    image = loader(image).unsqueeze(0)  # remove the fake batch dimension
    return image.to(device, torch.float)


def imshow(tensor: torch.Tensor,
           title: str = None) -> None:
    """
    Plot the image

    Parameters
    ----------
    tensor (torch.tensor):
        torch.Tensor with image
    title (string):
        Title of plot with image
    Returns
    -------
        None
    """
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
