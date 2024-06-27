import torch
import numpy as np
from PIL import Image
from .aura_sr import AuraSR

upscaler = None
class AuraSRNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image":("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "upscale"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_AuraSR"

    def upscale(self, image):
        global upscaler
        if upscaler is None:
            upscaler = AuraSR.from_pretrained()
        out_images = []
        for img_i in image.numpy():
            image_np = img_i * 255
            image_np = image_np.astype(np.uint8)
            out_image = upscaler.upscale_4x(Image.fromarray(image_np))
            out_image = np.array(out_image).astype(np.float32) / 255.0
            out_images.append(torch.from_numpy(out_image)[None,])
        
        output_images = torch.cat(out_images, dim=0)
        print(output_images.shape)
        return (output_images,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "AuraSRNode": AuraSRNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AuraSRNode": "AuraSRNode"
}

