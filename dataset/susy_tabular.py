import torch
import pandas as pd
from tqdm import tqdm
from dataset.Susy import prepare_dataset, build_feature_map_Susy

def convert_to_tabular_text_features(dataset, save_path="susy_test_text_features.csv"):
    rows = []

    for idx, data in tqdm(enumerate(dataset), total=len(dataset), desc="Processing dataset"):
        try:
            fmap = build_feature_map_Susy(data)
            feature_text = ", ".join(
                f"{name} is {data.x[row][col].item():.3f}"
                for name, (row, col) in fmap.items()
                if not torch.isnan(data.x[row][col])
            )
            label = int(data.y.item())
            label_str = "background" if label == 0 else "signal"

            rows.append({
                "features_text": feature_text,
                "label": label_str
            })
        except Exception as e:
            print(f"Skipping example {idx} due to error: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved {len(df)} rows to {save_path}")

def main():
    # Load only test set
    trainset, _, testset = prepare_dataset()
    convert_to_tabular_text_features(trainset, save_path="susy_train_text_features.csv")
    convert_to_tabular_text_features(testset, save_path="susy_test_text_features.csv")

if __name__ == "__main__":
    main()
