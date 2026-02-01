# inference.py for ActionCLIP model

from pathlib import Path
import json
import torch
import torchvision
from pathlib import Path
import json
from PIL import Image
import yaml
from dotmap import DotMap
from tqdm import tqdm
import pandas as pd
import os
from typing import List
import numpy as np
import sys
import gc
import cv2
import torchvision.transforms as T
import json
from pathlib import Path
import warnings
import torch.nn as nn
from torch.cuda.amp import autocast
from resources.ActionCLIP.datasets.transforms_ss import *
from resources.ActionCLIP.modules.Visual_Prompt import visual_prompt
from resources.ActionCLIP import clip
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


# Setting up a few parameters for testing
######################################################################################
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

#data = {"num_frames_to_sample": 16}
mixed_precision = False

print_cuda_information_model = False
save_first_frame = False
use_two_models = False
######################################################################################


print(f"\n\nINPUT_PATH: {INPUT_PATH}")
print(f"OUTPUT_PATH: {OUTPUT_PATH}")
print("-" * 20)
print(f"use_two_models: {use_two_models}")
print(f"mixed_precision: {mixed_precision}")
print("-" * 20)
print("-" * 20)
print(f"print_cuda_information_model: {print_cuda_information_model}")
print(f"save_first_frame: {save_first_frame}\n\n")


# utils/Text_Prompt.py
######################################################################################
def classes_all(config):
    classes_all = pd.read_csv(config.data.labels_file_path)
    
    return classes_all.values.tolist()

def class_ids_only(config):
    classes_all = pd.read_csv(config.data.labels_file_path)
    
    return classes_all['id'].tolist()
    

def text_prompt(config, use_anticipation=False):
    
    # normal labels: 'configs/label_map_t3.txt' # label_map_t3_subset
    # custom labels: 'configs/label_maps/t3_generated/label_map_t3_llama3-1.txt'
    
    # original, but adapt it for verb + noun pairs
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
        
    if use_anticipation:
        text_aug = [
                        f"a photo of the next action {{}}",
                        f"a picture of the next action {{}}",
                        f"Human action that will happen: {{}}",
                        f"{{}}, an upcoming action",
                        f"{{}}, this will be the next action",
                        f"{{}}, a video of the next action",
                        f"About to perform the action {{}}",
                        f"{{}}", 
                        f"About to do a kind of action, {{}}",
                        f"Preparing to do a kind of action, {{}}",
                        f"Look, the human is about to {{}}",
                        f"Can you recognize the next action of {{}}?",
                        f"Video anticipation of {{}}",
                        f"A video of the next action: {{}}",
                        f"The man will {{}}",
                        f"The woman is going to {{}}"
                    ]

    text_dict = {}
    text_dict_readible = {}
    num_text_aug = len(text_aug)
    classes = None

    print(f"Number of classes in labels file: {len(classes_all(config))}")
    for ii, txt in enumerate(text_aug):
            text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in classes_all(config)])
            text_dict_readible[ii] = [txt.format(c) for i, c in classes_all(config)]
    
    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict, text_dict_readible
######################################################################################


# utils/Augmentation.py
######################################################################################
def get_augmentation(training, config):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = config.data.input_size * 256 // 224
    if training:
        None
        '''
        unique = torchvision.transforms.Compose([GroupMultiScaleCrop(config.data.input_size, [1, .875, .75, .66]),
                                                 GroupRandomHorizontalFlip(is_sth='some' in config.data.dataset),
                                                 GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                                                        saturation=0.2, hue=0.1),
                                                 GroupRandomGrayscale(p=0.2),
                                                 GroupGaussianBlur(p=0.0),
                                                 GroupSolarization(p=0.0)]
                                                )
        '''
    else:
        unique = torchvision.transforms.Compose([GroupScale(scale_size),
                                                 GroupCenterCrop(config.data.input_size)])

    common = torchvision.transforms.Compose([Stack(roll=False),
                                             ToTorchFormatTensor(div=True),
                                             GroupNormalize(input_mean,
                                                            input_std)])
    return torchvision.transforms.Compose([unique, common])
######################################################################################


# train.py model wrapper (original)
######################################################################################
class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)
######################################################################################


# Load model
######################################################################################
def load_model(cache_dir, model_paths, config, frames_per_clip):
    """
    Loads a CLIP model and its variants for additional anticipation.

    Args:
        cache_dir (str): Directory path to store the model.
        model_paths (list): List of model paths to load.
        config (dict): Configuration dictionary.
        frames_per_clip (int): Number of frames per clip.

    Returns:
        tuple: A tuple containing the loaded models and the device used.
    """
    
    if use_two_models:
        # removed for ActionCLIP
        None
    
    else:
        model_path = model_paths[0]
        model_dir = str(Path(model_path).parent)
        
        print("\nModel Dir:")
        print(f"model_dir: {model_dir}")
        print("\n")
        
        # init models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=frames_per_clip, dropout=config.network.drop_out, # T=config.data.num_segments,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32
        
        
        # FOR additional anticipation: features of frames --> natural temporal information of current and next
        fusion_model = visual_prompt(config.network.sim_header,clip_state_dict,frames_per_clip) # ,config.data.num_segments
        fusion_model = torch.nn.DataParallel(fusion_model).cuda()
        
        # FOR additional anticipation: same image encoder for both tasks (same frame)
        model_image = ImageCLIP(model)
        model_image = torch.nn.DataParallel(model_image).cuda()
        
        # FOR additional anticipation: shared encoder (recognition & anticipation) learns richer representations that understand both 
        # --> current states and future predictions, text_embeddings are different for both tasks and only the underlying clip model gets updated (similar to DINOv2 and VideoMAE with same foundation)
        model_text = TextCLIP(model)
        model_text = torch.nn.DataParallel(model_text).cuda()

        if device == "cpu":
            model_text.float()
            model_image.float()
        else:
            clip.model.convert_weights(
                model_text) 
            clip.model.convert_weights(model_image)

        # Load pre-trained model
        if os.path.isfile(model_path):
            print(("=> loading checkpoint '{}'".format(model_path)))
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
            print(("=> Loaded checkpoints from '{}' successfully!".format(model_path)))
        else:
            print(("=> no checkpoint found at '{}'".format(model_path)))
            
        model.eval()
        fusion_model.eval()
        
        model_image.eval()
        model_text.eval()
        print("\n\n")
        
        models = (model, fusion_model, model_image, model_text)

    return models, device

######################################################################################


# Load data sample
######################################################################################
class DummyRecord:
    def __init__(self, start_frame=0, stop_frame=0, num_frames=0, path=""):
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.num_frames = num_frames
        self.path = path


def get_val_indices(record, num_segments, seg_length):
        """
        Get the valid frame indices for the given record and parameters.

        Parameters:
        record (DummyRecord): record containing start_frame, stop_frame, num_frames, and path
        num_segments (int): number of segments to sample from record
        seg_length (int): length of each segment

        Returns:
        np.array: array of valid frame indices
        """
        
        total_length = num_segments * seg_length
        loop = False
        
        start_frame = record.start_frame
        stop_frame = record.stop_frame
        bias = start_frame
            
        valid_indices = np.arange(record.num_frames) + bias

        if len(valid_indices) == 0: #not valid_indices:
            print("\nWARNING: No valid frames found for this record.")
            return np.array([])

        valid_indices_set = set(valid_indices)  # Faster lookup

        if num_segments == 1:
            chosen = valid_indices[len(valid_indices) // 2]
            return np.array([chosen], dtype=int)

        if record.num_frames <= total_length:
            if loop:
                sampled = np.mod(np.arange(total_length), record.num_frames) + bias
            else:
                sampled = np.array([i * record.num_frames // total_length
                                    for i in range(total_length)], dtype=int) + bias
        else:
            offset = (record.num_frames / num_segments - seg_length) / 2.0
            sampled = np.array([i * record.num_frames / num_segments + offset + j
                                for i in range(num_segments)
                                for j in range(seg_length)], dtype=int) + bias

        # Replace invalid indices with unique valid ones
        sampled_set = set(sampled)  # Track used indices
        final_sampled = []

        for idx in sampled:
            if idx in valid_indices_set:
                final_sampled.append(idx)
            else:
                # Choose a replacement from valid_indices that is not in final_sampled
                replacements = list(valid_indices_set - sampled_set)
                if replacements:
                    new_idx = np.random.choice(replacements)
                    final_sampled.append(new_idx)
                    sampled_set.add(new_idx)  # Mark as used
                else:
                    print("\nWARNING: Not enough valid indices to replace all invalid ones. Using duplicates.")
                    
                    # Fixed empty array error, when there arent enough valid frames, just use dupes
                    # Get the number of required frames
                    required_frames = num_segments * seg_length
                    
                    # Sample with replacement from the valid indices to fill the required length
                    final_sampled = np.random.choice(valid_indices, size=required_frames, replace=True)
    
                    return np.array(final_sampled, dtype=int)

        return np.array(final_sampled, dtype=int)


def calculate_uniform_frame_indices(start_frame: int, stop_frame: int, num_frames_to_sample: int) -> List[int]:
    """Calculate uniform frame indices for sampling"""
    
    total_frames = stop_frame - start_frame + 1
    
    if total_frames <= num_frames_to_sample:
        return list(range(start_frame, stop_frame + 1))
    
    indices = np.linspace(start_frame, stop_frame, num_frames_to_sample, dtype=int)
    return indices.tolist()


def extract_clip(video_path, start_frame, stop_frame, resize=None, config=None):
    """
    Extract frames between start and stop with smart uniform sampling.
    Optimized for speed by reading sequentially.
    """
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    sample_frame = None
    total_available_frames = stop_frame - start_frame + 1
    
    # use dummy to keep same structure for simplicity
    record = DummyRecord(
        start_frame=start_frame,
        stop_frame=stop_frame,
        num_frames=total_available_frames,
        path=video_path,
    )
    
    try:
        segment_indices = get_val_indices(record, config.data.num_segments, config.data.seg_length) # f.E. [116 147 178 209 241 272 303 334]
        if len(segment_indices) == 0:
            raise ValueError("\nNo valid segments found. Using default uniform sampling.")
        frame_indices_to_load = segment_indices
            
    except (ValueError, IndexError) as e:
        total_length = config.data.num_segments * config.data.seg_length
        frame_indices_to_load = set(calculate_uniform_frame_indices(start_frame, stop_frame, total_length)) # f.E. {101, 136, 172, 207, 243, 278, 314, 350}
    
    # Start reading from the beginning of the clip
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame_idx = start_frame
    
    while current_frame_idx <= stop_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if the current frame is one we want to load
        if current_frame_idx in frame_indices_to_load:
            
            # Store sample frame (first frame)
            if not sample_frame:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
                sample_frame = (pil_frame, current_frame_idx)
            
            # Convert from BGR (OpenCV) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize if requested
            if resize:
                frame = cv2.resize(frame, resize)

            pil_frame = Image.fromarray(frame).convert("RGB")
            frames.extend([pil_frame])
            
        current_frame_idx += 1
        
    cap.release()

    if not frames:
        return None, fps, None
    
    return frames, fps, sample_frame
######################################################################################


# ActionCLIP inference
######################################################################################
def encode_text_features(model, classes, classes_anticipation, device):
    """
    Encode ALL possible text templates for ALL classes upfront.
    
    Args:
        model: clip model
        classes: tensor containing all possible class labels
        classes_anticipation: tensor containing all possible class labels for anticipation
        device: device to put tensors on
    
    Returns:
        text_features: normalized text embeddings for recognition classes
        anticipation_text_features: normalized text embeddings for anticipation classes
    """
    
    # Encode ALL possible text templates for ALL classes upfront
    text_inputs = classes.to(device)
    text_features = model.encode_text(text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    anticipation_text_inputs = classes_anticipation.to(device)
    anticipation_text_features = model.encode_text(anticipation_text_inputs)
    anticipation_text_features = anticipation_text_features / anticipation_text_features.norm(dim=-1, keepdim=True)

    return text_features, anticipation_text_features


def run_inference_on_clip(model, fusion_model, model_image, model_text, 
                                clip_tensor, device, start_frame, stop_frame, 
                                text_features, anticipation_text_features, num_text_aug, num_text_aug_anticipation, #text_dict, text_dict_anticipation,
                                config, frames_per_clip):
    """
    Run inference on a single clip (video segment).

    Inputs:
    - model: CLIP model
    - fusion_model: fusion model (image-text fusion)
    - model_image: image model
    - model_text: text model
    - clip_tensor: input video tensor
    - device: device to run inference on (e.g. cuda:0)
    - start_frame: start frame of the clip
    - stop_frame: stop frame of the clip
    - text_features: pre-computed text embeddings for recognition classes
    - anticipation_text_features: pre-computed text embeddings for anticipation classes
    - num_text_aug: number of text augmentations for recognition
    - num_text_aug_anticipation: number of text augmentations for anticipation
    - config: configuration object
    - frames_per_clip: number of frames per clip

    Outputs:
    - dictionary containing recognition and anticipation outputs
    """
    
    with torch.inference_mode():
        clip_tensor_gpu = clip_tensor.to(device, dtype=torch.float32)
        
        c_nseg_seqL, h, w = clip_tensor_gpu.shape
        num_frames = c_nseg_seqL // 3  # infer number of frames dynamically
        if num_frames != frames_per_clip:
            print(f"num_frames={num_frames}, frames_per_clip={frames_per_clip}")
        
        if num_frames >= frames_per_clip:
            clip_tensor_gpu = clip_tensor_gpu.view((-1, frames_per_clip, 3) + clip_tensor_gpu.size()[-2:]) 
        else:
            # Fallback: use however many frames there are (if less frames than frames_per_clip specified in config)
            clip_tensor_gpu = clip_tensor_gpu.view((-1, num_frames, 3) + clip_tensor_gpu.size()[-2:]) 
        print(f"Clip {start_frame}-{stop_frame} -> permuted shape: {clip_tensor_gpu.shape}")
        
        b, t, c, h, w = clip_tensor_gpu.size() 
        image_input = clip_tensor_gpu.view(-1, c, h, w) 
        print(f"Clip {start_frame}-{stop_frame} -> input shape (B*T,..): {image_input.shape}")
        
        if mixed_precision:
            with autocast():
                image_features = model.encode_image(image_input).view(b, t, -1)
                image_features = fusion_model(image_features)
        else:
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        print(f"Clip {start_frame}-{stop_frame} -> image_embedding shape: {image_features.shape}")
                
        
        # Recognition
        
        # Compute cosine similarity (scaled by 100) between image and text features
        # Resulting shape: (b, num_classes * num_text_aug)
        # --> Compute similarity between each image and ALL text templates (templates with all classes)
        #text_features = text_features.to(device)
        similarity = (100.0 * image_features @ text_features.T) 
        
        # Reshape to separate text augmentations: (b, num_text_aug, num_classes)
        similarity = similarity.view(b, num_text_aug, -1)
        logits_recognition = similarity # f.E. torch.Size([1, 16, 124])
        
        # Convert similarities to probabilities via softmax across classes
        similarity = similarity.softmax(dim=-1)
        # Average predictions across text augmentations for each sample
        
        # Final shape: (b, num_classes)
        similarity = similarity.mean(dim=1, keepdim=False) # f.E. torch.Size([1, 124])
        
        
        # Anticipation
        
        #anticipation_text_features = anticipation_text_features.to(device)
        anticipation_similarity = (100.0 * image_features @ anticipation_text_features.T)
        anticipation_similarity = anticipation_similarity.view(b, num_text_aug_anticipation, -1)
        logits_anticipation = anticipation_similarity
        anticipation_similarity = anticipation_similarity.softmax(dim=-1)
        anticipation_similarity = anticipation_similarity.mean(dim=1, keepdim=False)


        if use_two_models:
            # removed for ActionCLIP
            None
        else:  
            # Recognition
            values_1, indices_1 = similarity.topk(1, dim=-1) # Shape: (b, 1) --> values_1 the highest probabilities per sample | indices_1 the predicted class indices (argmax)
            values_5, indices_5 = similarity.topk(5, dim=-1) # Shape: (b, 5) 
            
            # Anticipation
            anticipation_values_1, anticipation_indices_1 = anticipation_similarity.topk(1, dim=-1) 
            anticipation_values_5, anticipation_indices_5 = anticipation_similarity.topk(5, dim=-1)
            
            print(f"Clip {start_frame}-{stop_frame} -> similarity shape (ar): {similarity.shape}")
            return {
                'recognition': {
                    'logits': logits_recognition.cpu(),
                    'probabilities': similarity.cpu(), 
                    'prediction_class': indices_1.item(), 
                    'top5_prediction_class': indices_5.squeeze(0).cpu().tolist(),
                    'confidence': values_1.item(),
                    'top5_confidence': values_5.squeeze(0).cpu().tolist(),
                },
                'anticipation': {
                    'logits': logits_anticipation.cpu(),
                    'probabilities': anticipation_similarity.cpu(),
                    'prediction_class': anticipation_indices_1.item(),
                    'top5_prediction_class': anticipation_indices_5.squeeze(0).cpu().tolist(),
                    'confidence': anticipation_values_1.item(),
                    'top5_confidence': anticipation_values_5.squeeze(0).cpu().tolist(),
                }
            } 
######################################################################################


# Process test video clips
######################################################################################
def process_video(video_path: Path, frame_bounds_path: Path, cache_dir, model_paths, #data,
                          save_first_frame=False, output_dir=None, use_lower_resolution=False, video_resolution=(640, 360), word_dict_file=None,
                          config=None, frames_per_clip=8):
    """
    Process a video using a CLIP model.

    Parameters:
        video_path (Path): path to input video file
        frame_bounds_path (Path): path to JSON file containing frame bounds
        cache_dir (Path): path to cache directory
        model_paths (List[str]): paths to model checkpoint files
        data (Action_DATASETS): data object
        save_first_frame (bool): save the first frame of the video
        output_dir (Path): path to output directory
        use_lower_resolution (bool): use lower video resolution (320x180)
        video_resolution (Tuple[int, int]): video resolution (width, height)
        word_dict_file (Path): path to WordDict file
        config (Config): configuration object
        frames_per_clip (int): number of frames per clip

    Returns:
        List[Dict[str, Any]]: list of dictionaries containing recognition and anticipation results for each clip
        List[Dict[str, Any]]: list of dictionaries containing saved frames for each clip

    """
    
    print("\nLoading model...\n")
    models, device = load_model(cache_dir, model_paths, config, frames_per_clip)
    
    if use_two_models:
        # removed for ActionCLIP
        None
    else:
        model, fusion_model, model_image, model_text = models
        #model = models
        model_device = device
        
    # Create transforms
    transform = get_augmentation(False, config)
    
    # Load frame bounds
    frame_bounds = load_json_file(location=frame_bounds_path)
    print("Frame bounds loaded:", frame_bounds)
    
    reverse_dict = load_word_dict(word_dict_file)
    total_next_action = 0
    
    
    # Load text prompts
    print("\nLoading text prompts...\n")
    classes, num_text_aug, text_dict, text_dict_readable = text_prompt(config, use_anticipation=False)
    classes_anticipation, num_text_aug_anticipation, text_dict_anticipation, text_dict_readable_anticipation = text_prompt(config, use_anticipation=True)

    print(f"Loaded {len(classes)} classes from {config.data.labels_file_path}")
    print(f"Loaded {len(classes_anticipation)} classes_anticipation from {config.data.labels_file_path}")
    
    
    print("\nEncoding text prompts...\n")
    with torch.no_grad():
        if mixed_precision:
            with autocast():
                text_features, anticipation_text_features = encode_text_features(
                    model, classes, classes_anticipation, device
                )
                             
                text_features = text_features.half()
                anticipation_text_features = anticipation_text_features.half()
        else: 
            text_features, anticipation_text_features = encode_text_features(
                model, classes, classes_anticipation, device
            )
         
    print(f"recognition: text_embedding shape (ntemp*nclass,hdim): {text_features.shape}")
    print(f"anticipation: text_embedding shape (ntemp*nclass,hdim): {anticipation_text_features.shape}")   
            
    all_results = []
    stored_frames = []
    
    print(f"\nProcessing {len(frame_bounds)} clips...\n")
    with tqdm(total=len(frame_bounds), desc="Processing clips") as pbar:
        for clip_idx, bounds in enumerate(frame_bounds):
            start_frame = bounds["start_frame"]
            stop_frame = bounds["stop_frame"]
                
            # extract video clip
            frames, fps, sample_frame = extract_clip(video_path, start_frame, stop_frame, resize=None, config=config)
           
            if not frames:
                print(f"[Warning] No frames loaded for clip {start_frame}-{stop_frame}")
                pbar.update(1)
                
            # Apply transforms
            if transform:
                try:
                    clip_tensor = transform(frames)
                    print(f"\nClip {start_frame}-{stop_frame} -> transformed shape (C*nseg*seqL,H,W): {clip_tensor.shape}")
                except Exception as e:
                    
                    print(f"[Info] Resizing clip to 256x256 as fallback")
                    clip_tensor = torch.nn.functional.interpolate(
                        clip_tensor.unsqueeze(0).float(), size=(256, 256), mode="bilinear", align_corners=False
                    ).squeeze(0)
                    print(f"\n[Error] Transform failed for clip {start_frame}-{stop_frame}: {e}")
                    continue
                
            # IMMEDIATE inference
            try:
                if use_two_models:
                    # removed for ActionCLIP
                    None
                else:
                    result = run_inference_on_clip(model, fusion_model, model_image, model_text, clip_tensor, model_device, start_frame, stop_frame, 
                                                   text_features, anticipation_text_features, num_text_aug, num_text_aug_anticipation, #text_dict, text_dict_anticipation,
                                                   config, frames_per_clip)
                    
                    pred_class_id_recognition = result['recognition']['prediction_class']
                    pred_class_id_anticipation = result['anticipation']['prediction_class']

                    # Convert class index to action string
                    current_action = reverse_dict[pred_class_id_recognition]
                    
                    if clip_idx + 1 < len(frame_bounds):
                        next_action = reverse_dict[pred_class_id_anticipation]
                    else:
                        next_action = ""
                    
                    
                    # Print inference results for both heads (commented out to reduce output right now)
                    # Recognition
                    rec = result['recognition']
                    #print(f"Clip {start_frame}-{stop_frame} Recognition:")
                    #print(f"  Top-1 class: {rec['prediction_class']} (confidence: {rec['confidence']:.3f})")
                    #print(f"  Top-5 classes: {rec['top5_prediction_class']} (confidences: {[f'{v:.3f}' for v in rec['top5_confidence']]})")
                    #print(f"  Logits shape (pre softmax;mean): {rec['logits'].shape}, Probabilities shape (default similarity scores): {rec['probabilities'].shape}\n")

                    # Anticipation
                    ant = result['anticipation']
                    #print(f"Clip {start_frame}-{stop_frame} Anticipation:")
                    #print(f"  Top-1 class: {ant['prediction_class']} (confidence: {ant['confidence']:.3f})")
                    #print(f"  Top-5 classes: {ant['top5_prediction_class']} (confidences: {[f'{v:.3f}' for v in ant['top5_confidence']]})")
                    #print(f"  Logits shape (pre softmax;mean): {ant['logits'].shape}, Probabilities shape (default similarity scores): {ant['probabilities'].shape}\n")
                    
                    # Only top-1 for summary
                    print(f"Clip {start_frame}-{stop_frame} Top1-Pred -> Recognition: {rec['prediction_class']} (conf: {rec['confidence']:.3f}), "
                            f"Anticipation: {ant['prediction_class']} (conf: {ant['confidence']:.3f})\n")

                all_results.append({
                    "stop_frame": stop_frame,
                    "next_action": next_action,
                    "start_frame": start_frame,
                    "current_action": current_action,
                })
                
            except Exception as e:
                print(f"[Error] Inference failed for clip {start_frame}-{stop_frame}: {e}")
                all_results.append({
                    "stop_frame": stop_frame,
                    "next_action": "",
                    "start_frame": start_frame,
                    "current_action": "",
                })
            
            # Save first frame (fast using original PIL image)
            if save_first_frame and sample_frame:
                pil_frame, frame_num = sample_frame
                stored_frames.append({
                    "frame_number": frame_num,
                    "frame": pil_frame.copy()
                })

            # Clear memory
            del clip_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            pbar.update(1)
    
    # Print summary
    successful_clips = sum(1 for r in all_results if r.get('current_action') is not None)
    print(f"\nProcessing complete: {successful_clips}/{len(all_results)} clips processed successfully\n")
    
    return all_results, stored_frames

def save_life_saving_actions_gc(results, output_file, stored_frames):
    """Save results in Grand Challenge format and frames to output/temp"""
    
    output_dir = output_file.parent
    temp_dir = output_dir / "temp"  # store frames in output/temp
    
    # Ensure directories exist
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    predictions = []
    for r in results:
        predictions.append({
            "start_frame": r.get("start_frame", None),
            "stop_frame": r.get("stop_frame", None),
            "current_action": r.get("current_action", ""),
            "next_action": r.get("next_action", "")
        })
    
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
    
    print(f"Life-Saving Actions saved to {output_file}")
    
    for frame_info in stored_frames:
        frame_number = frame_info["frame_number"]
        pil_frame = frame_info["frame"]
        
        # Zero-pad to 5 digits: e.g., frame_00105.jpg
        filename = temp_dir / f"frame_{frame_number:05d}.png"
        pil_frame.save(filename, quality=85, optimize=True)

    print(f"{len(stored_frames)} frames saved to {temp_dir}")
    

# Additional methods
######################################################################################
def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())
    
def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

def load_file(*, location):
    # Reads the content of a file
    with open(location) as f:
        return f.read()

def load_annotations(annotation_file):
    with open(annotation_file, "r") as f:
        annotations = json.load(f)
    return annotations

def load_word_dict(word_dict_file):
    word_dict = load_json_file(location=word_dict_file)
    # Reverse dictionary: {class_number: action_name}
    reverse_dict = {v: k for k, v in word_dict["actions_dict"].items()}
    return reverse_dict

def _show_torch_cuda_info():
    print("=+=" * 10)
    print("\nCollecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
            
    print("=+=" * 10)
######################################################################################


# Main section to run the inference script
######################################################################################
def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface) --> simplified
    handler = {
        ("life-saving-procedure", "video-frame-bounds"): interf0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    # Read the input files using Grand Challenge paths
    video_path = Path(INPUT_PATH / "life-saving-procedure.mp4")
    frame_bounds_path = Path(INPUT_PATH / "video-frame-bounds.json")
    
    if Path("/opt/app/resources").exists():
        sys.path.append("/opt/app/resources") 
        RESOURCE_PATH = Path("/opt/app/resources")  # In container (not uing /model folder)
    else:
        sys.path.append("resources")
        RESOURCE_PATH = Path("resources")
    MODEL_PATH = Path(RESOURCE_PATH / "model")

    word_dict_file = Path(RESOURCE_PATH / "word_dict.json")
    
    cache_dir = Path(MODEL_PATH / "cache")
    model_dir_recognition = Path(MODEL_PATH / "recognition")
    model_path_recognition = Path(model_dir_recognition / "model_best.pt")
    model_dir_anticipation = Path(MODEL_PATH / "anticipation")
    model_path_anticipation = Path(model_dir_anticipation / "model_best.pt")
    
    model_dir_combined = Path(MODEL_PATH / "combined")
    model_path_combined = Path(model_dir_combined / "model_best.pt")
    
    model_paths = [model_path_combined]
    if use_two_models:
        model_paths = [model_path_recognition, model_path_anticipation]
    _show_torch_cuda_info()

    
    config_path = Path(RESOURCE_PATH / "t1_test_2025.yaml")
    with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    config = DotMap(config)
    config.data.labels_file_path = Path(config_path.parent) / config.data.label_list

    global frames_per_clip
    frames_per_clip = config.data.num_segments * config.data.seg_length # T # num_frames
    
    # Process video using existing pipeline
    results, stored_frames = process_video(
        video_path=video_path,
        frame_bounds_path=frame_bounds_path,
        cache_dir=cache_dir,
        model_paths=model_paths,
        output_dir=None,
        save_first_frame=save_first_frame,
        word_dict_file=word_dict_file,
        
        config=config,
        frames_per_clip=frames_per_clip,
    )

    # Save results in Grand Challenge format
    save_life_saving_actions_gc(results, OUTPUT_PATH / "life-saving-actions.json", stored_frames)
    
    return 0
######################################################################################

if __name__ == "__main__":
    raise SystemExit(run())
