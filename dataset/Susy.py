import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from sparticles import EventsDataset  # Make sure this path is correct in your project
import tqdm
import pandas as pd


def build_feature_map_Susy(data):
    """
    Builds a feature map based on the shape of data.x.
    Handles events with 6 or 7 rows representing different particle features.
    """
    if data.x.shape[0] == 7:
        return {
            "Transverse Momentum of jet 1": (0, 0), "Pseudorapidity of jet 1": (0, 1), "Azimuthal angle of jet 1": (0, 2), "Quantile of jet 1": (0, 3),
            "Transverse Momentum of jet 2": (1, 0), "Pseudorapidity of jet 2": (1, 1), "Azimuthal angle of jet 2": (1, 2), "Quantile of jet 2": (1, 3),
            "Transverse Momentum of jet 3": (2, 0), "Pseudorapidity of jet 3": (2, 1), "Azimuthal angle of jet 3": (2, 2), "Quantile of jet 3": (2, 3),
            "Transverse Momentum of bjet 1": (3, 0), "Pseudorapidity of bjet 1": (3, 1), "Azimuthal angle of bjet 1": (3, 2), "Quantile of bjet 1": (3, 3), "massa bjet 1": (3, 4),
            "Transverse Momentum of bjet 2": (4, 0), "Pseudorapidity of bjet 2": (4, 1), "Azimuthal angle of bjet 2": (4, 2), "Quantile of bjet 2": (4, 3), "massa bjet 2": (4, 4),
            "Transverse Momentum of lepton": (5, 0), "Pseudorapidity of lepton": (5, 1), "Azimuthal angle of lepton": (5, 2),
            "Missing transverse energy": (6, 0), "Azimuthal Angle of Missing Transverse Energy": (6, 2), "Missing Transverse Energy Significance": (6, 5)
        }
    elif data.x.shape[0] == 6:
        return {
            "Transverse Momentum of jet 1": (0, 0), "Pseudorapidity of jet 1": (0, 1), "Azimuthal angle of jet 1": (0, 2), "Quantile of jet 1": (0, 3),
            "Transverse Momentum of jet 2": (1, 0), "Pseudorapidity of jet 2": (1, 1), "Azimuthal angle of jet 2": (1, 2), "Quantile of jet 2": (1, 3),
            "Transverse Momentum of bjet 1": (2, 0), "Pseudorapidity of bjet 1": (2, 1), "Azimuthal angle of bjet 1": (2, 2), "Quantile of bjet 1": (2, 3), "massa bjet 1": (2, 4),
            "Transverse Momentum of bjet 2": (3, 0), "Pseudorapidity of bjet 2": (3, 1), "Azimuthal angle of bjet 2": (3, 2), "Quantile of bjet 2": (3, 3), "massa bjet 2": (3, 4),
            "Transverse Momentum of lepton": (4, 0), "Pseudorapidity of lepton": (4, 1), "Azimuthal angle of lepton": (4, 2),
            "Missing transverse energy": (5, 0), "Azimuthal Angle of Missing Transverse Energy": (5, 2), "Missing Transverse Energy Significance": (5, 5)
        }
    else:
        raise ValueError(f"Unexpected data.x shape: {data.x.shape}")

    

def compose_prompt(data, feature_map):
    """
    Composes a natural language prompt from event features and a task instruction.
    Returns a string to feed into an LLM for classification.
    """
    prompt_intro = (
        "You are a particle physicist tasked with classifying high-energy collision events from particle physics experiments. \n\n"
        "Each event is represented by a set of physics features, including jets, b-jets, leptons, and missing transverse energy (MET). "
        "Your goal is to analyze these features and determine the nature of the event.\n\n"
        "Your task is to classify the event as one of the following:\n"
        "- SUSY Signal (indicating supersymmetric particle production), or\n"
        "- Standard Model Background (such as QCD jets or top quarks).\n\n"
        "Here are the features of the current event that you need to classify:\n"
    )

    features_text = "\n".join([ 
        f"{name}: {data.x[row][col].item():.3f}" 
        for name, (row, col) in feature_map.items() 
        if not torch.isnan(data.x[row][col])  # skip nan values
    ])

    task_description = (
        "\nPlease provide your classification by responding with either \"signal\" or \"background\".\n"
        "Answer:\n"
    )

    return prompt_intro + features_text + task_description


def prepare_dataset():
    dataset = EventsDataset(
        root='~/Desktop',
        url='https://cernbox.cern.ch/s/0nh0g7VubM4ndoh/download',
        delete_raw_archive=False,
        add_edge_index=True,
        event_subsets={'signal': 100000, 'singletop': 50000, 'ttbar': 50000}
    )
    from sklearn.model_selection import train_test_split
    trainset, testset = train_test_split(dataset, test_size=0.2)
    trainset, evalset = train_test_split(trainset, test_size=0.2)
    return trainset, evalset, testset


def prepare_icl_prompt(trainset, num_signal=5, num_background=5):
    signal_examples = [e for e in trainset if e.y.item() == 1][:num_signal]
    background_examples = [e for e in trainset if e.y.item() == 0][:num_background]

    def format_example(event, label_text):
        fmap = build_feature_map_Susy(event)
        return compose_prompt(event, fmap).replace("Answer:\n", f"Answer: {label_text}\n")

    parts = [format_example(e, "signal") for e in signal_examples]
    parts += [format_example(e, "background") for e in background_examples]
    return "\n---\n".join(parts) + "\n---\n"