from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target: torch.Tensor = target.detach()

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """
        Calculate content loss

        Parameters
        ----------
        input (torch.Tensor):
            input tensor
        Returns
        -------
            torch.Tensor
        """
        self.loss: torch.Tensor = F.mse_loss(input, self.target)
        return input

def gram_matrix(input: torch.Tensor) -> torch.Tensor:
    """
    Calculate Gram matrix

    Parameters
    ----------
    input (torch.Tensor):
        input tensor
    Returns
    -------
        Gram matrix
    """
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target: torch.Tensor = gram_matrix(target_feature).detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Calculate Gram matrix

        Parameters
        ----------
        input (torch.Tensor):
            input tensor
        Returns
        -------
            torch.Tensor
        """
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input