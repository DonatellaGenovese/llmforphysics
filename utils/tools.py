import os
import json
import torch
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM , AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from pathlib import Path
from datasets import load_dataset
import csv
import random

def load_model(model_name: str, 
               use_flash_attention: bool = False, 
               device: str = "cuda", 
               torch_dtype=torch.float16, 
               quantization: bool = False, 
               resume_download: bool =True) -> (torch.nn.Module, AutoTokenizer):
    
    #fix configuration from pretrained model
    config = AutoConfig.from_pretrained(model_name)
    #fix tokenizer from pretrained model 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantization == True:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config
        )
        
    else:
        model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto"
        )
        
        
    if model_name == "meta-llama/Llama-3.1-8B-Instruct" or model_name == "meta-llama/Llama-3.1-8B" or model_name == "meta-llama/Llama-2-7b-hf" or model_name == "meta-llama/Llama-3.3-70B-Instruct":
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_flash_attention:
        if hasattr(model, "enable_flash_attention"):
            model.enable_flash_attention()
            print("Flash Attention enabled.")
        else:
            print("Flash Attention not supported for this model.")
    
    return model, tokenizer

def load_gemma3_model(model_name: str, 
                      use_flash_attention: bool = False, 
                      device: str = "cuda", 
                      torch_dtype=torch.bfloat16,
                      device_map: str = "auto") -> (torch.nn.Module, AutoProcessor):
    """
    Loads a Gemma3 model along with its processor.
    
    Parameters:
        model_name (str): The model identifier (e.g., "google/gemma-3-27b-it").
        use_flash_attention (bool): Whether to enable flash attention if available.
        device (str): The target device for the model (e.g., "cuda" or "cpu").
        torch_dtype: The torch data type to be used (default is torch.bfloat16).
        device_map (str): The device mapping strategy (default "auto").
    
    Returns:
        model (torch.nn.Module): The loaded Gemma3 model.
        processor (AutoProcessor): The corresponding processor for the model.
    """
    
    # Load the Gemma3 model with the specified device map and data type.
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name, 
        device_map=device_map,
        torch_dtype=torch_dtype
    ).eval()
    
    # Move the model to the specified device.
    model.to(device)
    
    # Optionally enable flash attention if supported.
    if use_flash_attention:
        if hasattr(model, "enable_flash_attention"):
            model.enable_flash_attention()
            print("Flash Attention enabled.")
        else:
            print("Flash Attention not supported for this model.")
    
    # Load the processor for handling inputs.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer


import torch

def generate_text_batch(model, tokenizer, prompts, max_length=30, device="cuda"):
    # Tokenize the prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generation settings
    generation_config = {
        "max_new_tokens": max_length,
        "temperature": 0.2,
        "do_sample": True,
        "return_dict_in_generate": True,
        "output_scores": False,
        "pad_token_id": tokenizer.eos_token_id
    }

    # Generate responses
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    generated_sequences = outputs.sequences
    full_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

    print('full_texts:', full_texts)

    responses = []

    for full_text in full_texts:
        main_response = ""
        additional_info = ""

        if "Answer:" in full_text:
            answer_part = full_text.rsplit("Answer:", 1)[-1].strip()
            parts = answer_part.split(maxsplit=1)

            if len(parts) >= 1:
                main_response = parts[0].strip()
            if len(parts) == 2:
                additional_info = parts[1].strip()

        else:
            parts = full_text.strip().split(maxsplit=1)
            if len(parts) >= 1:
                main_response = parts[0]
            if len(parts) == 2:
                additional_info = parts[1]

        responses.append({
            "main_response": main_response,
            "additional_info": additional_info
        })

    return responses




    
def generate_text(model, tokenizer, prompt, max_length=30, device="cuda"):
    
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    generation_config = {
        "max_new_tokens": max_length,
        "temperature": 0.3,
        "do_sample": True,
        "return_dict_in_generate": True,
        "output_scores": False
    }
    output = model.generate(input_ids=input_ids, **generation_config)
    # Extract the generated sequence from the dictionary
    generated_text = tokenizer.decode(output["sequences"][0], skip_special_tokens=True)

    return generated_text

    
    

def save_answers_csv(model_answers, original_labels, additional_info, output):
    file_exists = os.path.exists(output)
    with open(output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_response", "original_label", "additional_info"])
        
        if not file_exists:
            writer.writeheader()
        
        for model_answer, label, info in zip(model_answers, original_labels, additional_info):
            writer.writerow({
                "model_response": model_answer,
                "original_label": label,
                "additional_info": info
            })
    torch.cuda.empty_cache()
    
