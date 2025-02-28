import os
import json
import torch
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from pathlib import Path
from datasets import load_dataset
import csv
import random

def load_model(model_name: str, 
               use_flash_attention: bool = False, 
               device: str = "cuda", 
               torch_dtype=torch.float16) -> (torch.nn.Module, AutoTokenizer):
    """
    Loads a Hugging Face language model and tokenizer, and optionally enables Flash Attention.
    
    Args:
        model_name (str): The name or path of the pretrained model.
        use_flash_attention (bool): Whether to enable flash attention if available.
        device (str): The device to map the model to (e.g., 'cuda' or 'cpu').
        torch_dtype: The torch data type to use (e.g., torch.float16 for FP16).
    
    Returns:
        model (torch.nn.Module): The loaded language model.
        tokenizer (AutoTokenizer): The corresponding tokenizer.
    """
   
    config = AutoConfig.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=config,
        torch_dtype=torch_dtype,
        # device_map="auto"
    )

    model.to(device)
    
    if use_flash_attention:
        if hasattr(model, "enable_flash_attention"):
            model.enable_flash_attention()
            print("Flash Attention enabled.")
        else:
            print("Flash Attention not supported for this model.")
    
    return model, tokenizer

def generate_responses_for_popqa_batch(model, tokenizer, device, output_csv="gemma2b_popqa_responses.csv"):
    # Load the dataset and convert to a list for efficient sampling
    ds_split = load_dataset("akariasai/PopQA")['test']
    ds_list = list(ds_split)
    total_examples = len(ds_list)
    
    fieldnames = ["idx", "question", "possible_answers", "model_response"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process the dataset in batches of 100 examples
        batch_size_prompt = 10
        for batch_start in range(0, total_examples, batch_size_prompt):
            prompts = []
            metadata = []  # To store (idx, question, possible_answers)
            # Create prompt list for the current chunk
            for idx in range(batch_start, min(batch_start + batch_size_prompt, total_examples)):
                example = ds_list[idx]
                question = example["question"]
                possible_answers = (
                    " | ".join(example["possible_answers"]) 
                    if isinstance(example["possible_answers"], list) 
                    else example["possible_answers"]
                )
                # Select 10 random examples from the pre-converted list for context
                examples = random.sample(ds_list, 10)
                example_texts = "\n\n".join(
                    [
                        f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['possible_answers'].split(' | ')[0]}"
                        if isinstance(ex["possible_answers"], str)
                        else f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['possible_answers'][0]}"
                        for i, ex in enumerate(examples)
                    ]
                )
    
                prompt = f"""Answer the following questions concisely and accurately, using no more than one sentence. Follow the format of the examples provided.

{example_texts}

Now, answer the following:
Question: {question}
Answer:"""
                prompts.append(prompt)
                metadata.append((idx, question, possible_answers))
            
            # Process the current chunk using the list of prompt strings
            
            model_answers = generate_text_batch(prompts, max_new_tokens=100, batch_size=10)
    
            # Save the responses to CSV along with the associated metadata
            for (idx, question, possible_answers), model_answer in zip(metadata, model_answers):
                writer.writerow({
                    "idx": idx,
                    "question": question,
                    "possible_answers": possible_answers,
                    "model_response": model_answer
                })
            del prompts, metadata, model_answers
            torch.cuda.empty_cache()  # Free up cached GPU memory
            print(f"Finished processing batch starting at index {batch_start} to {min(batch_start + batch_size_prompt, total_examples)-1}")


def generate_text_batch(model, tokenizer, prompts, max_length=30, device="cuda"):
    # Properly tokenize and move tensors to device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to device

    generation_config = {
        "max_new_tokens": max_length,
        "temperature": 0,
        "do_sample": False,
        "return_dict_in_generate": True, 
        "output_scores": False
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    # Ensure we extract only `sequences` from GenerateOutput
    generated_sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs

    # Decode the generated sequences
    responses = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

    return responses
    
    

def generate_text(model, tokenizer, prompt, max_length=30, device="cuda"):
    
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    generation_config = {
        "max_new_tokens": max_length,
        "temperature": 0,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": False
    }
    output = model.generate(input_ids=input_ids, **generation_config)
    # Extract the generated sequence from the dictionary
    generated_text = tokenizer.decode(output["sequences"][0], skip_special_tokens=True)

    return generated_text

    
    

def save_answers_csv(metadata, model_answers, output):
    file_exists = os.path.exists(output)
    with open(output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "question", "possible_answers", "model_response"])
        if not file_exists:
            writer.writeheader()
        for idx, question, possible_answers, model_answer in zip(
            metadata["idx"], metadata["question"], metadata["possible_answers"], model_answers
        ):
            writer.writerow({
                "idx": idx,
                "question": question,
                "possible_answers": possible_answers,
                "model_response": model_answer
            })
    torch.cuda.empty_cache()