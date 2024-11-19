import argparse
import pandas as pd
import os
from tools.clustering import KMeans, KMeansParamsGrid
from tools.config import PREPROCESSED_DATA_DIR, CLUSTERED_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="The name of the dataset to run the model on", choices=["hepatitis", "mushroom", "synthetic"], required=True)
parser.add_argument("--model", type=str, help="The name of the model to run", choices=["kmeans"], required=True)

model_map = {
    "kmeans": KMeans,
}
params_grid_map = {
    "kmeans": KMeansParamsGrid,
}

def main():
    args = parser.parse_args()
    preprocessed_data_path = PREPROCESSED_DATA_DIR / args.dataset / f"{args.dataset}.csv"
    preprocessed_data = pd.read_csv(preprocessed_data_path)

    model = model_map[args.model]()

    model.fit(preprocessed_data)
    clusters = model.predict(preprocessed_data)
    clustered_data = pd.concat([preprocessed_data, pd.Series(clusters, name="cluster")], axis=1)

    os.makedirs(CLUSTERED_DATA_DIR / args.dataset, exist_ok=True)
    clustered_data_path = CLUSTERED_DATA_DIR / args.dataset / f"{args.dataset}_{args.model}.csv"
    clustered_data.to_csv(clustered_data_path, index=False)

if __name__ == "__main__":
    main()
