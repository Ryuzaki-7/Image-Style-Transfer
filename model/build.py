from model import metaArch

def build_model(cfg, pool='max'):
    model_factory = getattr(metaArch, cfg.MODEL.META_ARCHITECTURE)
    model = model_factory(cfg, pool)
    return model
