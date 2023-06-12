import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os

from detectron2.structures import ImageList

class ViTMatte(nn.Module):
    def __init__(self,
                 *,
                 backbone,
                 criterion,
                 pixel_mean,
                 pixel_std,
                 input_format,
                 size_divisibility,
                 decoder,
                 ):
        super(ViTMatte, self).__init__()
        self.backbone = backbone
        self.criterion = criterion
        self.input_format = input_format
        self.size_divisibility = size_divisibility
        self.decoder = decoder
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
    
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images, targets, H, W = self.preprocess_inputs(batched_inputs)

        # torch.save(images, 'images.pt')

        # from huggingface_hub import HfApi
        # api = HfApi()
        # api.upload_file(
        #     path_or_fileobj="images.pt",
        #     path_in_repo="images.pt",
        #     repo_id="nielsr/vitmatte-dummy-data",
        #     repo_type="dataset",
        # )

        features = self.backbone(images)

        print("Shape of backbone features:", features.shape)
        print("First values:", features[0,0,:3,:3])

        outputs = self.decoder(features, images)  

        if self.training:
            assert targets is not None
            trimap = images[:, 3:4]
            sample_map = torch.zeros_like(trimap)
            sample_map[trimap==0.5] = 1
            losses = self.criterion(sample_map ,outputs, targets)               
            return losses
        else:

            print("Shape of phas:", outputs['phas'].shape)
            print("First values of phas:", outputs['phas'][0,0,:3,:3])

            outputs['phas'] = outputs['phas'][:,:,:H,:W]
            return outputs



    def preprocess_inputs(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = batched_inputs["image"].to(self.device)
        trimap = batched_inputs['trimap'].to(self.device)
        images = (images - self.pixel_mean) / self.pixel_std

        if 'fg' in batched_inputs.keys():
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 1
            trimap[trimap >= 85] = 0.5

        images = torch.cat((images, trimap), dim=1)
        
        B, C, H, W = images.shape
        if images.shape[-1]%32!=0 or images.shape[-2]%32!=0:
            new_H = (32-images.shape[-2]%32) + H
            new_W = (32-images.shape[-1]%32) + W
            new_images = torch.zeros((images.shape[0], images.shape[1], new_H, new_W)).to(self.device)
            new_images[:,:,:H,:W] = images[:,:,:,:]
            del images
            images = new_images

        if "alpha" in batched_inputs:
            phas = batched_inputs["alpha"].to(self.device)
        else:
            phas = None

        return images, dict(phas=phas), H, W