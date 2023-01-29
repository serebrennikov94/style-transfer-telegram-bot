from __future__ import print_function

import torch
import torchvision.transforms as transforms
from io import BytesIO
from PIL.Image import Image

from slow_model.image_processing import image_loader
from slow_model.model_utils import run_style_transfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set available device
cnn = torch.load('./slow_model/model').features.to(device).eval()  # load pretrained model


def run_model(content_image: Image,
              style_image: Image) -> BytesIO:
    """
    Initialize style transfer process
    and get bytes of result image

    Parameters
    ----------
    content_image (Image):
        Content image in PIL format
    style_image (Image):
        Style image in PIL format
    Returns
    -------
        Bytes of result image after style transfer process
    """
    imsize = 512 if torch.cuda.is_available() else 300   # use small size if no gpu

    content_img = image_loader(content_image, imsize, device)
    style_img = image_loader(style_image, imsize, device)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    input_img = content_img.clone()

    output = run_style_transfer(
        cnn=cnn, normalization_mean=cnn_normalization_mean,
        normalization_std=cnn_normalization_std,
        content_img=content_img, style_img=style_img,
        input_img=input_img
    ).detach().to('cpu').squeeze(0)
    transform = transforms.ToPILImage()
    image = transform(output)
    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)
    return bio









