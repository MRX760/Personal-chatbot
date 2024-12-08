from huggingface_hub import InferenceClient
import json
import tempfile
import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler, UNet2DConditionModel
from langchain_core.prompts import PromptTemplate
from peft import LoraConfig, get_peft_model, PeftModel
import copy
from transformers import AutoTokenizer

import pandas as pd
from PIL import Image
import re
from io import StringIO
import PyPDF2
from docx import Document
import shutil

def save_log(messages, filename='log_dump.txt'):
    with open(filename, 'w') as txt:
        json.dump(messages, txt)

def read_log(filename='log_dump.txt'):
    with open(filename, 'r') as log:
        messages = json.load(log)
    return messages

def send_img_request(prompt, negative_prompt,
                     model_path='..\stable-diffusion-webui\models\Stable-diffusion', 
                     custom_weights='realisticVisionV60B1_v51HyperVAE.safetensors', 
                     inference_steps=6,
                     cfg_scale=2,
                     use_karras = True,
                     seed=0, use_lora=True,
                     width=512, height=512):
    
    custom_weights = os.path.join(model_path, custom_weights)
    
    #clear vram
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print(f'Running on GPU {torch.cuda.get_device_name(0)}')
        device="cuda"
    else:
        print('No cuda found, running on CPU')
        device="cpu"
    
    #opening custom weight or checkpoints
    pipe = StableDiffusionPipeline.from_single_file(custom_weights, torch_dtype=torch.float16).to(device)
    
    if use_lora:        
        lora_dir = '..\stable-diffusion-webui\models\Lora'
        lora_file = ['add_detail.safetensors', 'neon_palette_offset.safetensors', 'more_details.safetensors']
        lora_weights = [1.0, 1.0, 0.5]
        lora_adapters = [i.split(".")[0] for i in lora_file]
        
        #iterate each lora weights
        for ldir, la in zip(lora_file, lora_adapters):
            print(ldir, la)
            pipe.load_lora_weights(lora_dir, weight_name=ldir,adapter_name=la)
        pipe.set_adapters(lora_adapters, adapter_weights=lora_weights)

    #scheduler DPM SDE++ Karras
    print("scheduling")
    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config,
                                                              use_karras_sigmas = use_karras)
    
    #seed assignment
    torch.manual_seed(seed=seed)

    #generating image
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=inference_steps, guidance_scale=cfg_scale, width=width, height=height).images[0]
    print(image.size)
    return image

def ask_img(prompt):
    # Create a regex pattern to match the specified structure
    # This will look for 'keyword' followed by 'image' somewhere in the string
    pattern = r'\b(craft|design|generate|create|draw|illustrate|render|make)\b(?:\s+\w+){0,5}\s+\b(potrait|portrait|image|picture|pic|pics|img|photo|art|painting|paintings|design|potrayal|portrayal)\b'
    return bool(re.search(pattern, prompt, re.IGNORECASE))

def update_data(dataframe: pd.DataFrame, dir: str) -> pd.DataFrame:
    # if value == []:
    #     raise ValueError("Trying to add empty value to dataframe")
    try:
        # dataframe.loc[len(dataframe)] = value
        dataframe = pd.DataFrame({"Filename":os.listdir(dir), "checkbox":[True]*len(os.listdir(dir))})    
        return dataframe
    except ValueError:
        raise ValueError(f"Extra value on list. Data should be equal to column dataframe which are {len(dataframe.shape[1])}. List are {len(value)} index long")
    except Exception as e:
        print(f"Encounter an error while trying to add value to dataframe. {e}")

def create_session_dir():
    return tempfile.TemporaryDirectory()

def create_session_log(dir: str):
    log = os.path.join(dir, 'log.txt')
    with open(log, 'w') as file:
        file.write(json.dumps({})) #dictionary
    return log

def update_session_log(session_log: dict, log_path: str):
    with open(log_path, 'w') as log:
        log.write(json.dumps(session_log))


def save_session_file(file, name: str, dir_name: str):
    try:
        with open(os.path.join(dir_name, name), 'wb') as uf:
            uf.write(file)
        print("Session file saved..")
    
    except Exception as e:
        print(f"Error saving session file.. {e}")

def save_to_database(filename: str, filedir: str):
    try:
        file = os.path.join(filedir, filename)
        shutil.copy(file, './data')
        return "Requested file successfully saved to database. Table will be updated when you refresh the browser 🤖"
    except FileNotFoundError:
        return "Error saving file to database. File not found"
    except Exception as e:
        # Handle any other exceptions and return the error message
        return f"An error occurred: {e}" 
    
def read_session_file(filename: str, dir: str) -> str:
    try:
        
        filename = os.path.join(filename, dir)

        if filename.endswith('.csv') or filename.endswith('.xlsx'):
                content = pd.read_csv(filename)

        elif filename.endswith('.txt'):
            content = StringIO(filename.getvalue().decode("utf-8"))
            
        elif filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(filename)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                content += page.extract_text()
            
        elif filename.endswith('docx'):
            doc = Document(filename)
            for para in doc.paragraphs:
                content += para.text + "\n"
        
        return content
    
    except Exception as e:
        print("Error has occured.. {e}")

# def tokenizer(model_name: str, prompt: str) -> int:
#     # model_name = "meta-llama/Meta-Llama-3-70B-Instruct" 
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return tokenizer
#     # tokens = tokenizer.encode(prompt, add_special_tokens=True)
#     # return len(tokens)


