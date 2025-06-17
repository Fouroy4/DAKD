from __future__ import absolute_import

from torchvision.transforms import *

#from PIL import Image
import random
import math
import numpy as np
#import torch
import torch
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms

import torch
import numpy as np
import random
import cv2
from PIL import Image
import torchvision.transforms as transforms


class Cutout:
    """
    Cutout: Randomly set certain regions of the image to zero
    
    Reference:
    [8] DeVries, T., & Taylor, G. W. (2017). 
    Improved regularization of convolutional neural networks with cutout.
    arXiv preprint arXiv:1708.04552.
    """
    def __init__(self, alpha=0.001, fixed_region=None):
        """
        Initialize Cutout
        
        Args:
            alpha: Probability of setting pixels to zero
            fixed_region: Fixed region (x, y, width, height), if None then randomly select
        """
        self.alpha = alpha
        self.fixed_region = fixed_region
        
    def __call__(self, img):
        """
        Apply Cutout
        
        Args:
            img: Input image tensor [C, H, W]
            
        Returns:
            img_cutout: Image after applying Cutout
            mask: Mask indicating which pixels are kept (1) or set to zero (0)
        """
        img_copy = img.clone()
        C, H, W = img.shape
        
        if self.fixed_region is None:
            # Random mask: set pixels to zero with probability alpha
            mask = torch.FloatTensor(H, W).uniform_() > self.alpha
            mask = mask.expand_as(img)
            img_copy = img_copy * mask
            
            # Find a connected region to represent the cutout area (for visualization)
            # Convert probability mask to binary image
            binary_mask = mask[0].numpy().astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour as the region
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                region = (x, y, w, h)
            else:
                region = None
        else:
            # Use fixed region
            x, y, w, h = self.fixed_region
            region = self.fixed_region
            
            # Create mask
            mask = torch.ones_like(img)
            mask[:, y:y+h, x:x+w] = 0
            img_copy = img_copy * mask
        
        return img_copy, region


class CutMix(object):
    """
    Cross-modal CutMix: Replace part of RGB image region with corresponding region from IR image
    Requires both RGB and IR images as input
    """
    def __init__(self, probability=0.5, alpha=0.7):
        self.probability = probability
        self.alpha = alpha  # Coefficient controlling region size
    
    def __call__(self, rgb_img, ir_img=None):
        # If IR image is not provided, raise error
        if ir_img is None:
            raise ValueError("CutMix requires both RGB and IR images")
            
        # Probability check
        if random.uniform(0, 1) >= self.probability:
            return rgb_img
            
        # Select region size and position
        h, w = rgb_img.shape[1], rgb_img.shape[2]
        
        # Region size based on image size ratio, simulated with Gaussian distribution
        size_ratio = max(0.1, min(0.9, np.random.normal(self.alpha, 0.3)))
        region_h = int(h * size_ratio)
        region_w = int(w * size_ratio)
        
        # Randomly select center point
        center_y = random.randint(0, h)
        center_x = random.randint(0, w)
        
        # Calculate region boundaries
        x1 = max(0, center_x - region_w // 2)
        y1 = max(0, center_y - region_h // 2)
        x2 = min(w, x1 + region_w)
        y2 = min(h, y1 + region_h)
        
        # Mix images: keep RGB image but use IR image in selected region
        mixed_img = rgb_img.clone()
        mixed_img[:, y1:y2, x1:x2] = ir_img[:, y1:y2, x1:x2]
        
        return mixed_img


class MixUp(object):
    """
    Cross-modal MixUp: Blend RGB and IR images
    Linearly mix two images with specified ratio
    """
    def __init__(self, probability=0.5, alpha=1.2):
        self.probability = probability
        self.alpha = alpha  # Beta distribution parameter
    
    def __call__(self, rgb_img, ir_img=None):
        # If IR image is not provided, raise error
        if ir_img is None:
            raise ValueError("MixUp requires both RGB and IR images")
            
        # Probability check
        if random.uniform(0, 1) >= self.probability:
            return rgb_img
            
        # Sample mixing ratio from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 0.5  # Fixed mixing ratio
            
        # Limit lam range to avoid extreme mixing ratios
        lam = max(0.2, min(0.8, lam))
        
        # Linearly mix two images
        mixed_img = lam * rgb_img + (1 - lam) * ir_img
        
        return mixed_img




class CutSwap(object):
    def __init__(self, min_alpha=0.3, max_alpha=0.6, prob=1.0):
        """
        Initialize the CutBlur class.
        Args:
            min_alpha: Minimum proportion of the cut region size.
            max_alpha: Maximum proportion of the cut region size.
            prob: Probability of applying the CutBlur augmentation.
        """
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.prob = prob  # Probability of applying the augmentation

    def channel_exchange(self, img):
        """
        Apply Channel Exchange to an RGB image in (C, H, W) format.
        Args:
            img: Input image in (C, H, W) format (torch.Tensor).
        Returns:
            img: Image with channel exchange applied (torch.Tensor).
        """
        idx = random.randint(0, 2)  # 0 for R, 1 for G, 2 for B

        # Avoid multiple clones; directly modify img in-place
        if idx == 0:
            img[1, :, :].copy_(img[0, :, :])  # Set G channel to R
            img[2, :, :].copy_(img[0, :, :])  # Set B channel to R
        elif idx == 1:
            img[0, :, :].copy_(img[1, :, :])  # Set R channel to G
            img[2, :, :].copy_(img[1, :, :])  # Set B channel to G
        elif idx == 2:
            img[0, :, :].copy_(img[2, :, :])  # Set R channel to B
            img[1, :, :].copy_(img[2, :, :])  # Set G channel to B

        return img

    def cutblur(self, im1, im2):
        """
        Apply CutBlur operation, copying a random region from im2 to im1.
        Args:
            im1: Original RGB image in (C, H, W) format (torch.Tensor).
            im2: Exchanged image in (C, H, W) format (torch.Tensor).
        Returns:
            im1: Image after CutBlur is applied (torch.Tensor).
        """
        # Get height and width
        h, w = im1.shape[1], im1.shape[2]
        
        # Calculate independent random ratios for the height and width
        cut_ratio_h = torch.rand(1).item() * (self.max_alpha - self.min_alpha) + self.min_alpha  # Random ratio for height
        cut_ratio_w = torch.rand(1).item() * (self.max_alpha - self.min_alpha) + self.min_alpha  # Random ratio for width

        # Calculate the cut height and cut width based on the random ratios
        ch = min(int(h * cut_ratio_h), h)
        cw = min(int(w * cut_ratio_w), w)

        # If the cut area is 0, skip cutblur
        if ch == 0 or cw == 0:
            return im1  # No cutblur, return original image

        # Randomly select position for the cut region, ensuring it fits inside the image
        cy = random.randint(0, h - ch)
        cx = random.randint(0, w - cw)

        # Use in-place operation to copy the random region from im2 to im1
        im1[:, cy:cy+ch, cx:cx+cw].copy_(im2[:, cy:cy+ch, cx:cx+cw])

        return im1

    def __call__(self, img):
        """
        Apply both Channel Exchange and CutBlur to an image based on the given probability.
        Args:
            img: Input RGB image in (C, H, W) format (torch.Tensor).
        Returns:
            img_blurred: Image with Channel Exchange and CutBlur applied (torch.Tensor).
        """
        # Check if augmentation should be applied
        if torch.rand(1).item() < self.prob:
            # Apply channel exchange
            img_exchanged = self.channel_exchange(img)

            # Apply cutblur (copy random region from exchanged image to the original)
            img_blurred = self.cutblur(img, img_exchanged)

            return img_blurred
        else:
            # If augmentation is not applied, return the original image
            return img
        
class ChannelAdap(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5):
        self.probability = probability

       
    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
            # return img

        idx = random.randint(0, 3)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            img = img

        return img
        
        
class ChannelAdapGray(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5):
        self.probability = probability

       
    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
            # return img

        idx = random.randint(0, 3)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            if random.uniform(0, 1) > self.probability:
                # return img
                img = img
            else:
                tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:]
                img[0,:,:] = tmp_img
                img[1,:,:] = tmp_img
                img[2,:,:] = tmp_img
        return img

class ChannelRandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img
    


class PatchMixAugment:
    def __init__(self, patch_size=16, rgb_to_ir_ratio=0.5, num_images=4):
        self.patch_size = patch_size
        self.rgb_to_ir_ratio = rgb_to_ir_ratio  # Ratio of RGB to IR patches
        self.num_images = num_images  # Number of RGB and IR images to select for mixing

    def patchmix(self, img1_batch, img2_batch):
        """
        Apply PatchMix augmentation to the batch of RGB and IR images.
        Args:
            img1_batch: [batch_size, C, H, W] -> RGB images
            img2_batch: [batch_size, C, H, W] -> IR images

        Returns:
            A batch of mixed images [batch_size, C, H, W].
        """
        batch_size, C, H, W = img1_batch.size()
        patches_per_row = W // self.patch_size  # Number of patches per row
        patches_per_col = H // self.patch_size  # Number of patches per column

        mixed_images = []

        # Assuming 4 identities per batch, 8 images per identity
        images_per_identity = batch_size // 4  # 8 images per identity
        for i in range(0, batch_size, images_per_identity):
            # Process each identity's 8 images
            rgb_images = img1_batch[i:i+images_per_identity]  # 8 RGB images for identity
            ir_images = img2_batch[i:i+images_per_identity]   # 8 IR images for identity

            for j in range(images_per_identity):
                rgb_img = rgb_images[j]  # Single RGB image
                ir_img = ir_images[j]    # Single IR image

                # Initialize an empty tensor for the mixed image
                mixed_image = torch.zeros_like(rgb_img)

                # Row-wise patch mixing
                for row in range(patches_per_col):
                    patch_indices = list(range(patches_per_row))  # Patches in each row
                    random.shuffle(patch_indices)  # Shuffle patch positions for randomness

                    # Calculate how many RGB patches to select based on the ratio
                    num_rgb_patches = int(self.rgb_to_ir_ratio * patches_per_row)

                    for idx, patch_idx in enumerate(patch_indices):
                        if idx < num_rgb_patches:
                            selected_img = rgb_img  # Select from RGB image
                        else:
                            selected_img = ir_img  # Select from IR image

                        # Extract the patch
                        row_start = row * self.patch_size
                        row_end = (row + 1) * self.patch_size
                        col_start = patch_idx * self.patch_size
                        col_end = (patch_idx + 1) * self.patch_size

                        # Get the patch from the selected image
                        patch = selected_img[:, row_start:row_end, col_start:col_end]

                        # Place the patch in the mixed image
                        mixed_image[:, row_start:row_end, col_start:col_end] = patch

                # Append the mixed image to the result list
                mixed_images.append(mixed_image)

        # Return the mixed images as a tensor
        return torch.stack(mixed_images)  # Return a batch of mixed images
    
    def save_and_show_image(self, img_tensor, index):
        """Save and display the image tensor."""
        # De-normalize the image if you normalized it before
        denormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]
        )
        img_tensor = denormalize(img_tensor)

        # Clip the values to ensure they are in the valid range [0, 1] for visualization
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # Convert tensor to a NumPy array for visualization
        img_numpy = img_tensor.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]

        # Show the image using matplotlib
        plt.imshow(img_numpy)
        plt.title(f'Mixed Image {index}')
        plt.axis('off')
        plt.show()

        # Save the image if you want to save it
        filename = os.path.join("/home/lunet/coyz2/passion/mixed_imgs2/", f'img1_1_{index}.png')
        plt.imsave(filename, img_numpy)
