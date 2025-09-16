import torch
import pandas as pd
from tqdm import tqdm
from Susy import prepare_dataset, build_feature_map_Susy

def convert_to_structured_text_features(dataset, save_path="susy_test_structured_features.csv"):
    rows = []

    for idx, data in tqdm(enumerate(dataset), total=len(dataset), desc="Processing dataset"):
        try:
            # Build the feature map dynamically
            fmap = build_feature_map_Susy(data)

            # Format the structured text using the feature map
            feature_text = ""  # Removed the "Classify the following event?" part
            
            # Adding a check for missing columns before accessing them
            def get_feature_value(fmap_key, default_value="N/A"):
                if fmap_key in fmap:
                    return f"{data.x[fmap[fmap_key]]:.3f}"
                return default_value

            # Jet 1
            feature_text += "- **Jet 1**:\n"
            feature_text += f"  - Transverse Momentum: {get_feature_value('Transverse Momentum of jet 1')}\n"
            feature_text += f"  - Pseudorapidity: {get_feature_value('Pseudorapidity of jet 1')}\n"
            feature_text += f"  - Azimuthal Angle: {get_feature_value('Azimuthal angle of jet 1')}\n"
            feature_text += f"  - Quantile: {get_feature_value('Quantile of jet 1')}\n"

            # Jet 2
            feature_text += "- **Jet 2**:\n"
            feature_text += f"  - Transverse Momentum: {get_feature_value('Transverse Momentum of jet 2')}\n"
            feature_text += f"  - Pseudorapidity: {get_feature_value('Pseudorapidity of jet 2')}\n"
            feature_text += f"  - Azimuthal Angle: {get_feature_value('Azimuthal angle of jet 2')}\n"
            feature_text += f"  - Quantile: {get_feature_value('Quantile of jet 2')}\n"

            # Jet 3 (if available)
            if "Transverse Momentum of jet 3" in fmap:
                feature_text += "- **Jet 3**:\n"
                feature_text += f"  - Transverse Momentum: {get_feature_value('Transverse Momentum of jet 3')}\n"
                feature_text += f"  - Pseudorapidity: {get_feature_value('Pseudorapidity of jet 3')}\n"
                feature_text += f"  - Azimuthal Angle: {get_feature_value('Azimuthal angle of jet 3')}\n"
                feature_text += f"  - Quantile: {get_feature_value('Quantile of jet 3')}\n"

            # b-Jet 1
            feature_text += "- **b-Jet 1**:\n"
            feature_text += f"  - Transverse Momentum: {get_feature_value('Transverse Momentum of bjet 1')}\n"
            feature_text += f"  - Pseudorapidity: {get_feature_value('Pseudorapidity of bjet 1')}\n"
            feature_text += f"  - Azimuthal Angle: {get_feature_value('Azimuthal angle of bjet 1')}\n"
            feature_text += f"  - Quantile: {get_feature_value('Quantile of bjet 1')}\n"
            feature_text += f"  - Mass: {get_feature_value('massa bjet 1')}\n"

            # b-Jet 2
            feature_text += "- **b-Jet 2**:\n"
            feature_text += f"  - Transverse Momentum: {get_feature_value('Transverse Momentum of bjet 2')}\n"
            feature_text += f"  - Pseudorapidity: {get_feature_value('Pseudorapidity of bjet 2')}\n"
            feature_text += f"  - Azimuthal Angle: {get_feature_value('Azimuthal angle of bjet 2')}\n"
            feature_text += f"  - Quantile: {get_feature_value('Quantile of bjet 2')}\n"
            feature_text += f"  - Mass: {get_feature_value('massa bjet 2')}\n"

            # Lepton
            feature_text += "- **Lepton**:\n"
            feature_text += f"  - Transverse Momentum: {get_feature_value('Transverse Momentum of lepton')}\n"
            feature_text += f"  - Pseudorapidity: {get_feature_value('Pseudorapidity of lepton')}\n"
            feature_text += f"  - Azimuthal Angle: {get_feature_value('Azimuthal angle of lepton')}\n"

            # Missing Transverse Energy
            feature_text += "- **Missing Transverse Energy**:\n"
            feature_text += f"  - Value: {get_feature_value('Missing transverse energy')}\n"
            feature_text += f"  - Azimuthal Angle: {get_feature_value('Azimuthal Angle of Missing Transverse Energy')}\n"
            feature_text += f"  - Significance: {get_feature_value('Missing Transverse Energy Significance')}\n"

            # Get the label
            label = int(data.y.item())
            label_str = "background" if label == 0 else "signal"

            rows.append({
                "features_text": feature_text,
                "label": label_str
            })
        except KeyError as e:
            print(f"KeyError in example {idx}: {e}")
        except Exception as e:
            print(f"Skipping example {idx} due to unexpected error: {e}")

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved {len(df)} rows to {save_path}")

def main():
    # Load only test set
    trainset, _, testset = prepare_dataset()
    print(f"Trainset size: {len(trainset)}")
    print(f"Testset size: {len(testset)}")
    convert_to_structured_text_features(trainset, save_path="susy_train_structured_features.csv")
    convert_to_structured_text_features(testset, save_path="susy_test_structured_features.csv")

if __name__ == "__main__":
    main()
