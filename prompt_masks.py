from copy import deepcopy
import torch
from PIL import Image
import gradio as gr

class PromptSAM(object):
  
    def __init__(self, max_num_mask):
        self.masked_images = None
        self.IMG_FEAT = None
        self.max_mask_num = max_num_mask
    
    def config(self, model, tokenizer, preprocess, mask_generator):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.mask_generator = mask_generator
  
    def get_masked_images(self, img, MAX_NUM_MASK):
        masks = self.mask_generator.generate(img)
        masked_images = []
        for mask in masks[:MAX_NUM_MASK]:
            temp_img = deepcopy(img)
            temp_mask = mask['segmentation']
            temp_img[temp_mask == False] = [255,255,255]
            masked_images.append(temp_img)
        return masked_images

    def get_masked_features(self, masked_images, progress=gr.Progress()):
        IMG_FEAT = None
        first=True
        for masked_img in progress.tqdm(masked_images):
            image = self.preprocess(Image.fromarray(masked_img)).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if first:
                    IMG_FEAT = image_features
                    first=False
                else:
                    IMG_FEAT = torch.vstack([IMG_FEAT,image_features])
        return IMG_FEAT
                                  
    def upload_image(self, image_uploaded, progress=gr.Progress()):
        self.masked_images = self.get_masked_images(image_uploaded,self.max_mask_num)
        self.IMG_FEAT = self.get_masked_features(self.masked_images,progress)
        message = "Processing Done! Ready for Prompting."
        return message

    def prompt_sam(self, text):
        text = self.tokenizer([text])
        TEXT_FEAT = self.model.encode_text(text)
        TEXT_FEAT /= TEXT_FEAT.norm(dim=-1, keepdim=True)
        min_idx = torch.norm(self.IMG_FEAT - TEXT_FEAT , dim=1).argmin()
        return self.masked_images[min_idx]
