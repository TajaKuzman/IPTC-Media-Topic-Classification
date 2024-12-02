
import argparse
import pandas as pd
import json
import numpy as np
import os
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import gc
import torch
from tqdm.autonotebook import tqdm as notebook_tqdm
import wandb
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("current_sample", help="Name of the current sample: 20k, 15k, 10k, 5k, 2.5k, 1k")
    args = parser.parse_args()
    
current_sample = args.current_sample

print(current_sample)

# Open the dataset
df = pd.read_json("MaCoCu-main-multilingual-predicted-dataset.jsonl", orient="records", lines=True)

# Prepare the dataset into train and dev, and each should have two columns: text and labels.

df.rename(columns={"IPTC_pred":"labels"}, inplace=True)

df = df[["text", "labels", "split", f"{current_sample}_sample"]]

# Keep only samples that a part of the current sample
train_df = df[df[f"{current_sample}_sample"] == "yes"]

train_df = train_df[["text", "labels"]]

dev_df = df[df["split"] == "dev"]
dev_df = dev_df[["text", "labels"]]

print("Train and dev size:")
print(train_df.shape, dev_df.shape)

LABELS = train_df.labels.unique().tolist()

# Login to wandb
wandb.login()
wandb.init(settings=dict(init_timeout=120), project="IPTC")

# Open an empty list if there are no previous results
#results_hyperparameter_search = []

# %%
# If the file for results has already been created, open it

with open(f"results/hyperparameter-search-{current_sample}-results.json", "r") as results_file:
	results_hyperparameter_search = json.load(results_file)

# After getting initial feeling for the optimal epochs, run the model again on a selection of epochs to see what micro and macro F1 scores we obtain

lr = 8e-6

hyp_name = "epoch"
epoch_set = [24, 24, 26, 26]


model_args = ClassificationArgs()

# define hyperparameters
model_args ={"overwrite_output_dir": True,
             "labels_list": LABELS,
             "learning_rate": lr,
             "train_batch_size": 32,
             # Comment out no_cache and no_save if you want to save the model
             "no_cache": True,
             "no_save": True,
            # Only the trained model will be saved (if you want to save it)
            # - to prevent filling all of the space
            # "save_model_every_epoch":False,
             "max_seq_length": 512,
             "save_steps": -1,
            # Use these parameters if you want to evaluate during training
            #"evaluate_during_training": True,
            ## Calculate how many steps will each epoch have
            # num steps in epoch = training samples / batch size
            # Then evaluate after every 3rd epoch
            #"evaluate_during_training_steps": len(train_df.text)/32*3,
            #"evaluate_during_training_verbose": True,
            #"use_cached_eval_features": True,
            #'reprocess_input_data': True,
            "wandb_project": "IPTC",
            "use_multiprocessing":False,
            "use_multiprocessing_for_evaluation":False,
            "silent": True,
             }

for epoch in epoch_set:
	model_args["num_train_epochs"] = epoch

	# Define the model
	roberta_large_model = ClassificationModel(
	"xlmroberta", "xlm-roberta-large",
	num_labels=len(LABELS),
	use_cuda=True,
	args= model_args
	)

	print(f"Training started - epoch num {epoch}.")
	print(f"Size of training data: {train_df.shape}")

	# Train the model on train data
	roberta_large_model.train_model(train_df)
	# Evaluate te model on dev data

	# Get the true labels from the dev
	y_true = dev_df.labels

	y_pred = []

	pred_counter = 0

	for text in dev_df.text.tolist():
		pred_counter += 1
		if pred_counter%100 == 0:
			print("Prediction of text", pred_counter)
		individual_y_pred = roberta_large_model.predict([text])[0]
		y_pred.append(individual_y_pred[0])

	# Calculate the scores
	macro = f1_score(y_true, y_pred, labels=LABELS, average="macro")
	micro = f1_score(y_true, y_pred, labels=LABELS,  average="micro")
	print(f"Macro f1: {macro:0.3}, Micro f1: {micro:0.3}")

	# Save the results:
	rezdict = {
	"epoch":epoch,
	"learning_rate": 8e-6,
	"microF1": micro,
	"macroF1": macro,
	}

	results_hyperparameter_search.append(rezdict)

# Save the results
with open(f"results/hyperparameter-search-{current_sample}-results.json", "w") as res_file:
	json.dump(results_hyperparameter_search, res_file)

print(pd.DataFrame(results_hyperparameter_search).sort_values(by="macroF1", ascending=False).to_markdown(index=False))


gc.collect()
torch.cuda.empty_cache()