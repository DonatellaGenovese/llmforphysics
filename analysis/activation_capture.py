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
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Batch




sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.tools import load_model, load_gemma3_model, generate_text_batch, save_answers_csv
from dataset.Susy import build_feature_map_Susy, compose_prompt
from sparticles import EventsDataset
from utils.patch_utils import InspectOutput, parse_layer_idx  # your hook-based class
from utils.save_activations import combine_activations  

@torch.inference_mode() # Disables gradient computation for better efficiency
@hydra.main(config_path="../configs", config_name="config", version_base=None) # Load configuration file
def main(cfg: DictConfig):
    print('model name',cfg.model.model_name)
    
    seed = cfg.seed if "seed" in cfg else 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load model and tokenizer
    if cfg.model.model_name == "google/gemma-3-27b-it":
        model, tokenizer = load_gemma3_model(cfg.model.model_name, use_flash_attention=cfg.model.use_flash_attention, device=cfg.device)
    
    else:
        model, tokenizer = load_model(cfg.model.model_name, use_flash_attention=cfg.model.use_flash_attention, device=cfg.device, resume_download=True, quantization = False)
        
        
    #print('configurazione:',cfg)
    # Load dataset and create dataloader
    dataset = EventsDataset(
    root='~/Desktop',
    url='https://cernbox.cern.ch/s/0nh0g7VubM4ndoh/download',
    delete_raw_archive=False,
    add_edge_index=True,
    event_subsets={'signal': 10000, 'singletop': 5000, 'ttbar': 5000},
    download_type=1
    )
    trainset, testset = train_test_split(dataset, test_size = 0.2)
    trainset, evalset = train_test_split(trainset, test_size = 0.2)
    
    train_loader = GeoDataLoader(trainset, shuffle=True, batch_size=16)
    eval_loader  = GeoDataLoader(evalset, shuffle=True, batch_size=16)
    test_loader  = GeoDataLoader(testset, shuffle=False, batch_size=16)
    
    # Set up directory for saving answers and activations
    model_name = cfg.model.model_name.split("/")[-1]
    data_name = cfg.dataset.dataset_name
    print('data_name',data_name)
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
        for bid, data in enumerate(test_loader):
            individual_graphs = data.to_data_list() 
            for i, graph in enumerate(individual_graphs):
                feature_map = build_feature_map_Susy(graph)  # ✅ Now safe
                prompt = compose_prompt(graph, feature_map)

                print(f"\nPrompt for event {bid}-{i}:\n{prompt}\n")

                with InspectOutput(model, module_names, move_to_cpu=True) as inspector:
                     responses = generate_text_batch(model, tokenizer, [prompt], max_length=200)

                for response in responses:
                    label = graph.y.item()
                    save_answers_csv(
                    [response['main_response']],
                    [label],
                    [response['additional_info']],
                    [prompt],
                    save_dir / f"answers_{model_name}.csv"
                 )
                print(f"Graph {bid}-{i} Classification: {response['main_response']}")
                print(f"Details: {response['additional_info']}\n{'-'*50}")

            #for module, ac in inspector.catcher.items():
            #    layer_idx = parse_layer_idx(module)
            #    fname = f"layer{layer_idx}-id{bid}.pt"
            #    if "mlp" in module:
            #        torch.save(ac[0].float(), mlp_save_dir / fname)
            #    elif "self_attn" in module:
            #        torch.save(ac[0].float(), attn_save_dir / fname)
            #    else:
            #        torch.save(ac[0].float(), hidden_save_dir / fname)

        # After processing all batches, combine individual activation files into one tensor per layer.
        #combine_activations(save_dir, target_layers=list(range(num_layer)),
        #                    activation_type=activation_type,
        #                    analyse_activation_list=["mlp", "attn", "hidden"])

    
if __name__ == '__main__':
    main()

