import os
import torch

import timm

from icrt.util.args import VisionEncoderConfig
from icrt.util.model_constructor import vision_encoder_constructor

# timms = timm.list_models()#module="vision_encoder")
# print(timms)

vision_cfg = VisionEncoderConfig()
vision_cfg.vision_encoder = "resnet34"
# vision_cfg.num_classes = 0
# vision_cfg.pretained = True

# vision_cfg = timm.data.resolve_model_data_config(vision_cfg)
vision_encoder = vision_encoder_constructor(vision_cfg)
# vision_encoder.vision_encoder =
print(vision_encoder)

input_dummy = torch.randn(1, 10, 2, 3, 224, 224)
output = vision_encoder(input_dummy)
print(output.shape)
