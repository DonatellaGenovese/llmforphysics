import os
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.tools import load_model, load_gemma3_model, generate_text_batch, save_answers_csv
from dataset.popQA import PopQADataset 
from utils.patch_utils import InspectOutput, parse_layer_idx  # your hook-based class
from utils.save_activations import combine_activations  


@torch.inference_mode() # Disables gradient computation for better efficiency
@hydra.main(config_path="../configs", config_name="config") # Load configuration file
def main(cfg: DictConfig):
    # Load model and tokenizer
    if cfg.model.model_name == "google/gemma-3-27b-it":
        model, tokenizer = load_gemma3_model(cfg.model.model_name, use_flash_attention=cfg.model.use_flash_attention, device=cfg.device)
    
    else:
        model, tokenizer = load_model(cfg.model.model_name, use_flash_attention=cfg.model.use_flash_attention, device=cfg.device, quantization = False)
        
    # Load dataset and create dataloader
    if cfg.dataset.dataset_name == "popQA":
        dataset = PopQADataset(tokenizer, split=cfg.dataset.split, max_length=cfg.dataset.max_length)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.dataset.batch_size,
                            num_workers=cfg.dataset.num_workers)
    
    # Set up directory for saving answers and activations
    model_name = cfg.model.model_name.split("/")[-1]
    data_name = cfg.dataset.dataset_name
    save_dir = Path("cached_data") / model_name / data_name
    os.makedirs(save_dir, exist_ok=True)

    # Define activation type (e.g., "conflict" or "none_conflict")
    activation_type = 'conflict'
    
    # Create subdirectories for activations (hidden, mlp, self-attn)
    hidden_save_dir = save_dir / "activation_hidden" / activation_type
    mlp_save_dir    = save_dir / "activation_mlp"    / activation_type
    attn_save_dir   = save_dir / "activation_attn"   / activation_type
    for d in [hidden_save_dir, mlp_save_dir, attn_save_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Build list of module names to hook (for each target layer, add base, self-attn, and mlp
    num_layer = cfg.model.layers
    if hasattr(model, "language_model"):
        base = "language_model.model"
    else:
        base = "model"
    module_names = []
    for idx in range(num_layer):
        module_names.append(f"{base}.layers.{idx}")
        module_names.append(f"{base}.layers.{idx}.self_attn")
        module_names.append(f"{base}.layers.{idx}.mlp")
    
    model.eval()
    # Process each batch: generate responses and capture/save activations
    with torch.no_grad():
        for bid, batch in enumerate(dataloader):
            prompts = "Mi dai la definizione di Quantum Computing"
            metadata = batch["metadata"]
            
            with InspectOutput(model, module_names, move_to_cpu=True, last_position=False) as inspector:
                model_answers = generate_text_batch(model, tokenizer, prompts, max_length=3000)

            # Save generated responses (answers) to CSV
            save_answers_csv(metadata, model_answers, model_name)
            print("Answer: ", model_answers)

            # Now iterate through the captured activations.
            # The captured activation for each module is a tensor of shape [batch_size, seq_length, hidden_dim]
            for module, ac in inspector.catcher.items():
                # Instead of indexing by prompt_last_positions, capture the full sequence.
                ac_full = ac.float()  # ac_full has shape [batch_size, seq_length, hidden_dim]

                # For simplicity, if batch_size == 1, we use the first (and only) example.
                # Otherwise, you might want to save each example separately.
                layer_idx = parse_layer_idx(module)
                save_name = f"layer{layer_idx}-id{bid}.pt"
                if "mlp" in module:
                    torch.save(ac_full[0], mlp_save_dir / save_name)
                elif "self_attn" in module:
                    torch.save(ac_full[0], attn_save_dir / save_name)
                else:
                    torch.save(ac_full[0], hidden_save_dir / save_name)

        # After processing all batches, combine individual activation files into one tensor per layer.
        #combine_activations(save_dir, target_layers=list(range(num_layer)),
        #                    activation_type=activation_type,
        #                    analyse_activation_list=["mlp", "attn", "hidden"])

    
if __name__ == '__main__':
    main()

