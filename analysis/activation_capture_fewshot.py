import os
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from torch_geometric.data import Batch
import random
import numpy as np
import pandas as pd 
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.tools import load_model, load_gemma3_model, generate_text_batch, save_answers_csv
from dataset.Susy import build_feature_map_Susy, compose_prompt, prepare_dataset
from sparticles import EventsDataset
from utils.patch_utils import InspectOutput, parse_layer_idx
from utils.save_activations import combine_activations  
from dataset.Susy import prepare_dataset, prepare_icl_prompt

@torch.inference_mode()
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print('model name', cfg.model.model_name)

    # Set random seeds for reproducibility
    seed = cfg.seed if "seed" in cfg else 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the model
    if cfg.model.model_name == "google/gemma-3-27b-it":
        model, tokenizer = load_gemma3_model(cfg.model.model_name, use_flash_attention=cfg.model.use_flash_attention, device=cfg.device)
    else:
        model, tokenizer = load_model(cfg.model.model_name, use_flash_attention=cfg.model.use_flash_attention, device=cfg.device, resume_download=True, quantization=False)

    print(cfg.device)

    # Load the preformatted dataset
    train_df = pd.read_csv("/hdd3/dongen/Desktop/llmforphysics/dataset/susy_train_structured_features.csv")
    test_df = pd.read_csv("/hdd3/dongen/Desktop/llmforphysics/dataset/susy_test_structured_features.csv")

    # Select the first 10 examples from the trainset for in-context learning
    in_context_examples = train_df[:10]

    # Prebuild the context part for all examples (only once)
    context_prompts = "\n".join(
        f"{row['features_text']}\nAnswer: {row['label']}" for _, row in in_context_examples.iterrows()
    )

    # Prompt introduction
    prompt_intro = (
        "You are a particle physicist analyzing high-energy collision events from a particle physics experiment.\n"
        "Each event is represented by physics features describing jets, b-jets, leptons, and missing transverse energy (MET).\n\n"
        "Your task is to classify this event. Is it more likely to be:\n\n"
        "- SUSY Signal (supersymmetric particle production), or\n"
        "- Standard Model Background (e.g., QCD jets, top quarks)\n"
        "Respond with only one word: either signal or background.\n\n"
    )

    # Prepare save directories
    model_name = cfg.model.model_name.split("/")[-1]
    data_name = cfg.dataset.dataset_name
    save_dir = Path("cached_data") / model_name / data_name
    os.makedirs(save_dir, exist_ok=True)

    # Process the test set in batches
    batch_size = 8
    for batch_start in tqdm(range(0, len(test_df), batch_size), desc="Processing test set"):
        batch_end = min(batch_start + batch_size, len(test_df))
        batch_data = test_df.iloc[batch_start:batch_end]

        # Construct prompts for all examples in the batch
        prompts = [
            f"{prompt_intro}{context_prompts}\n\n{row['features_text']}\nRespond with only one word: signal or background."
            for _, row in batch_data.iterrows()
        ]

        # Generate responses for the entire batch
        responses = generate_text_batch(model, tokenizer, prompts, max_length=100)

        # Save the responses along with the labels
        for idx, response in enumerate(responses):
            test_example = batch_data.iloc[idx]
            label = test_example['label']
            save_answers_csv(
                [response['main_response']],
                [label],
                [response['additional_info']],
                save_dir / f"answers_{model_name}.csv"
            )
          #print(f"Test example {batch_start + idx}: Classification: {response['main_response']}")
            #print(f"Details: {response['additional_info']}\n{'-'*50}")

            # Optionally save activations
            # for module, ac in inspector.catcher.items():
            #     layer_idx = parse_layer_idx(module)
            #     fname = f"layer{layer_idx}-id{bid}.pt"
            #     if "mlp" in module:
            #         torch.save(ac[0].float(), mlp_save_dir / fname)
            #     elif "self_attn" in module:
            #         torch.save(ac[0].float(), attn_save_dir / fname)
            #     else:
            #         torch.save(ac[0].float(), hidden_save_dir / fname)

        # Optional: Combine activations
        # combine_activations(save_dir, target_layers=list(range(num_layer)),
        #                     activation_type=activation_type,
        #                     analyse_activation_list=["mlp", "attn", "hidden"])

if __name__ == '__main__':
    main()
