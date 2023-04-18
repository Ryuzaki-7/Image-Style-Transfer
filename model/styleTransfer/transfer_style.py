from torch.autograd import Variable
from util.imageTransform import ImageTransform
from util.utils import transform_image, optimize
from util.logger import setup_logger

logger = setup_logger('style-transfer', False)

def do_transfer_style(cfg, model, content_image,style_image,device):
    logger.info("Start transferring.")
    image_transformer = ImageTransform(cfg.DATA.IMG_SIZE, cfg.DATA.IMAGENET_MEAN)
    content_image = transform_image(image_transformer, content_image, device)
    style_image = transform_image(image_transformer, style_image, device)
    optimized_image = Variable(content_image.data.clone(), requires_grad=True)
    optimized_image = optimize(model, content_image, style_image, optimized_image, cfg, cfg.LOSS.MAX_ITER)
    out_image = image_transformer.post_preparation(optimized_image.data[0].cpu().squeeze())
    out_image.save(cfg.OUTPUT.DIR + cfg.OUTPUT.FILE_NAME)
    return out_image
