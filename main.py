import torch.nn as nn
import os
import torch
from PIL import Image
from model import build_model
from model.styleTransfer import do_transfer_style
from util.config import get_cfg_defaults
from util.lossFn import GramMSELoss

input_image = input("Enter the path for content: ")
st_image = input("Enter the path for style: ")

class StyleTransfer:
    def __init__(self,  vgg_model, loss_layers, loss_functions, loss_weights):
        self.vgg_model = vgg_model
        self.loss_layers = loss_layers
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights

def get_model(cfg):
    vgg_model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    vgg_model.to(device)
    vgg_model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))
    for param in vgg_model.parameters():
        param.requires_grad = False

    # layers, loss functions
    loss_layers = cfg.LOSS.STYLE_LAYERS + cfg.LOSS.CONTENT_LAYERS
    loss_functions = [GramMSELoss()] * len(cfg.LOSS.STYLE_LAYERS) + \
                     [nn.MSELoss()] * len(cfg.LOSS.CONTENT_LAYERS)
    loss_functions = [loss_function.to(device) for loss_function in loss_functions]
    loss_weights = cfg.LOSS.STYLE_WEIGHTS + cfg.LOSS.CONTENT_WEIGHTS
    model = StyleTransfer(vgg_model, loss_layers, loss_functions, loss_weights)
    return model, device

def transfer_style(cfg):
    model, device = get_model(cfg)
    content_image = Image.open(input_image)
    style_image = Image.open(st_image)
    do_transfer_style(cfg,model,content_image,style_image,device)

def main():
    cfg = get_cfg_defaults()
    cfg.freeze()
    if not os.path.isdir(cfg.OUTPUT.DIR):
        os.mkdir(cfg.OUTPUT.DIR)
    transfer_style(cfg)

if __name__ == "__main__":
    main()