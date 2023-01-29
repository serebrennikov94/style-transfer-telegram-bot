from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

from torch.optim import LBFGS

from slow_model.loss_utils import ContentLoss, StyleLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean: torch.Tensor = torch.tensor(mean).view(-1, 1, 1)
        self.std: torch.Tensor = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor with image

        Parameters
        ----------
        img (torch.Tensor):
            Tensor with image
        Returns
        -------
            Normalized image
        """
        # normalize img
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn,
                               normalization_mean: torch.Tensor,
                               normalization_std: torch.Tensor,
                               style_img: torch.Tensor,
                               content_img: torch.Tensor,
                               content_layers: List,
                               style_layers: List) -> Tuple:
    """
    Initialize model, style and content losses

    Parameters
    ----------
    cnn:
        Pytorch pretrained model
    normalization_mean (torch.Tensor):
        Tensor with mean value for each dimension
    normalization_std (torch.Tensor):
        Tensor with standard deviation value for each dimension
    style_img (torch.Tensor):
        torch.Tensor with style image
    content_img (torch.Tensor):
        torch.Tensor with content image
    content_layers (List):
        List of content layers
    style_layers (List):
        List of style layers
    Returns
    -------
        Tuple with model and lists of style and content losses
    """
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img: torch.Tensor) -> LBFGS:
    """
    Initialize optimizer for input tensor with image

    Parameters
    ----------
    input_img (torch.Tensor):
        Input torch.Tensor with image
    Returns
    -------
        optimizer
    """
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn,
                       normalization_mean: torch.Tensor,
                       normalization_std: torch.Tensor,
                       content_img: torch.Tensor,
                       style_img: torch.Tensor,
                       input_img: torch.Tensor,
                       style_weight: int = 100000,
                       content_weight: int = 1,
                       num_steps: int = 300) -> torch.Tensor:
    """
    Run style transfer process

    Parameters
    ----------
    cnn:
        Pytorch pretrained model
    normalization_mean (torch.Tensor):
        Tensor with mean value for each dimension
    normalization_std (torch.Tensor):
        Tensor with standard deviation value for each dimension
    content_img (torch.Tensor):
        torch.Tensor with content image
    style_img (torch.Tensor):
        torch.Tensor with style image
    input_img (torch.Tensor):
        torch.Tensor with input image
    style_weight (int):
        Weight of style image in result
    content_weight (int):
        Weight of content image in result
    num_steps (int):
        Amount of iterations for style transfer process.
        The more, the better.
    Returns
    -------
        Result image after style transfer process
    """
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img,
        content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        )

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img