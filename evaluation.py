import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import pingouin as pg

def calculate_metrics(ground_truths, predictions):
    """Calculate evaluation metrics."""
    mae = mean_absolute_error(ground_truths, predictions)
    r2 = r2_score(ground_truths, predictions)
    ratings = predictions + ground_truths 
    raters = [0] * len(predictions) + [1] * len(ground_truths)
    targets = list(range(len(predictions))) + list(range(len(ground_truths)))
    results_df = pd.DataFrame({"rating": ratings, "rater": raters, "ground_truth": targets})
    icc = pg.intraclass_corr(data=results_df, targets='ground_truth', raters='rater', ratings='rating')['ICC'].values[5]

    return mae, r2, icc

def main(args):
    context_scope = args.context_scope
    model_id = args.model_id  
    prompt_type = args.prompt_type
    madrs_item = args.madrs_item
    directory = args.directory

    ground_truth_file = os.path.join(directory, "full_dataset_processed.csv")
    ground_truth_df = pd.read_csv(ground_truth_file)
    scores = []
    for run in range(1, 6):  # Assuming 5 runs
        predictions = {}
        ground_truths = {}
        for patient_dir in os.listdir(directory):
            patient_dir = os.path.join(directory, patient_dir)
            if os.path.isdir(patient_dir):
                for session_dir in os.listdir(patient_dir):
                    session_path = os.path.join(patient_dir, session_dir)
                    if os.path.isdir(session_path):
                        predictions_dir = os.path.join(session_path, f"{context_scope}_{model_id}_{prompt_type}")
                        predictions_file = os.path.join(predictions_dir, f"madrs{madrs_item}_output_{run}.txt")
                        
                        if os.path.exists(predictions_file):
                            with open(predictions_file, "r") as f:
                                predicted_rating = int(f.read().split("Rating:")[-1].strip())
                            
                            session_id = f"{patient_dir}/{session_dir}"
                            predictions[session_id] = predicted_rating

                            ground_truth_rating = ground_truth_df[ground_truth_df["video_id"] == session_dir][f"madrs{madrs_item}"].values[0]

                            ground_truths[session_id] = ground_truth_rating
                            
        
        # Convert predictions and ground truths to lists for metric calculation
        y_true = [ground_truths[session_id] for session_id in predictions]
        predictions_list = [predictions[session_id] for session_id in predictions]

        mae, r2, icc = calculate_metrics(y_true, predictions_list)
        scoring = {
            "mae": mae,
            "r2": r2,
            "icc": icc
        }
        scores.append(scoring)
    
    # Aggregate scores from all runs and report mean +- standard deviation
    mean_mae = np.mean([score["mae"] for score in scores])
    mean_r2 = np.mean([score["r2"] for score in scores])
    mean_icc = np.mean([score["icc"] for score in scores])
    std_mae = np.std([score["mae"] for score in scores])
    std_r2 = np.std([score["r2"] for score in scores])
    std_icc = np.std([score["icc"] for score in scores])

    print(f"Model: {model_id}")
    print(f"Prompt Type: {prompt_type}")  
    print(f"MADRS Item: {madrs_item}")
    print(f"MAE: {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"R2: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"ICC: {mean_icc:.4f} ± {std_icc:.4f}")
    print("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier")
    parser.add_argument("--context_scope", type=str, required=True, help="Context scope (holistic vs segmented)")
    parser.add_argument("--prompt_type", type=str, required=True, help="Prompt type (e.g., 'full', 'no_desc', etc.)")
    parser.add_argument("--madrs_item", type=int, required=True, help="MADRS item number (0-10) or 11 for all items")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing CAMI data")
    args = parser.parse_args()

    if args.madrs_item < 0 or args.madrs_item > 11:
        raise ValueError("MADRS item must be between 0 and 11 (inclusive).")
    
    if args.madrs_item == 11:
        for i in range(1, 11):
            # Run the evaluation for each MADRS item from 1 to 10
            args.madrs_item = i
            main(args)
    else:
        # Run the evaluation for a single MADRS item
        main(args)