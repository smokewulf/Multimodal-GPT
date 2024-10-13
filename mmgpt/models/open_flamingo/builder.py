import open_clip
import torch
import torch.nn as nn
from bigmodelvis import Visualization
from peft import LoraConfig, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer

def create_model_and_transforms(
    vision_encoder,
    lang_encoder,
    eoc_token_id,
    media_token_id,
    vis_dim,
    cross_attn_every_n_layers=1,
    use_media_placement_augmentation=False,
    **flamingo_kwargs
):
    # Initialize the vision encoder
    vision_model = open_clip.create_model(vision_encoder, pretrained=True)
    
    # Initialize the language model
    lang_model = LlamaForCausalLM.from_pretrained(lang_encoder)
    tokenizer = LlamaTokenizer.from_pretrained(lang_encoder)
    
    # Configure the model with LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1, bias="none"
    )
    lang_model = get_peft_model(lang_model, lora_config)
    
    # Create a combined model
    class CombinedModel(nn.Module):
        def __init__(self, vision_model, lang_model):
            super(CombinedModel, self).__init__()
            self.vision_model = vision_model
            self.lang_model = lang_model

        def forward(self, images, input_ids):
            vision_features = self.vision_model(images)
            outputs = self.lang_model(input_ids=input_ids, encoder_hidden_states=vision_features)
            return outputs

    model = CombinedModel(vision_model, lang_model)
    
    # Visualization setup
    Visualization(model)
    
    return model, tokenizer
