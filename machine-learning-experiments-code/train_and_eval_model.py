import os
import argparse
import pandas as pd
import json
import numpy as np
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Name of the model.")
    parser.add_argument("current_sample", help="Name of the current sample: '20k_sample', '15k_sample', '10k_sample', '5k_sample', '2.5k_sample', '1k_sample', '5k_balanced_sample', '15k_sample_2', '15k_sample_3', '10k_sample_2', '10k_sample_3', '5k_sample_2', '5k_sample_3', '2.5k_sample_2', '2.5k_sample_3', '1k_sample_2', '1k_sample_3'")
    args = parser.parse_args()
    

model_name = args.model_name
current_sample = args.current_sample

print(current_sample)

# Train and save the model

# Open the training dataset
df = pd.read_json("MaCoCu-main-multilingual-predicted-dataset.jsonl", orient="records", lines=True)

df.rename(columns={"IPTC_pred":"labels"}, inplace=True)

df = df[["text", "labels", "split", f"{current_sample}"]]

# Keep only samples that are a part of the current sample
train_df = df[df[f"{current_sample}"] == "yes"]

print("Train split opened. Train data size:")
print(train_df.shape)

print(train_df["labels"].value_counts().to_markdown())

# Create a list of labels
LABELS = train_df.labels.unique().tolist()

# Login to wandb
wandb.login()
wandb.init(settings=dict(init_timeout=120), project="IPTC")

# Specify the hyperparameters
lr = 8e-06

if "20k_sample" in current_sample:
	epoch = 3
elif "15k_sample" in current_sample:
	epoch = 5
elif "10k_sample" in current_sample:
	epoch = 9
elif current_sample in ['5k_sample','5k_sample_2', '5k_sample_3', "5k_sample_4", "5k_sample_5"]:
	epoch = 10
elif current_sample == "5k_balanced":
	epoch = 10
elif "2.5k_sample" in current_sample:
	epoch = 22
elif "1k_sample" in current_sample:
	epoch = 24

print(f"epoch number: {epoch}")

model_args = ClassificationArgs()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# define hyperparameters
model_args ={"overwrite_output_dir": True,
             "labels_list": LABELS,
             "num_train_epochs": epoch,
             "learning_rate": lr, 
             "train_batch_size": 32,
             "output_dir": model_name, 
             # Comment out no_cache and no_save if you want to save the model
             #"no_cache": True,
             #"no_save": True,
            # Only the trained model will be saved (if you want to save it)
            # - to prevent filling all of the space
             "save_model_every_epoch":False,
             "max_seq_length": 512,
             "save_steps": -1,
            "wandb_project": "IPTC",
            "use_multiprocessing":False,
            "use_multiprocessing_for_evaluation":False,
            "silent": True,
             }


# Define the model
roberta_large_model = ClassificationModel(
"xlmroberta", "xlm-roberta-large",
num_labels=len(LABELS),
use_cuda=True,
args= model_args
)

print("Training started.")

# Train the model on train data
roberta_large_model.train_model(train_df)

print("Training completed.")

print("Evaluation started.")

# Evaluate the model

# Open the test sample
test_df = pd.read_json("MaCoCu-manually-annotated-test-set-with-preds.json", orient="records", lines=True)

print("Test file opened.")
print(test_df.shape)

print(test_df.head(1).to_dict())

# Open the main results file:
previous_results_file = open("results/BERT-model-experiments-results.json")
previous_results = json.load(previous_results_file)

# Calculate the model's predictions on test
def make_prediction(input_string):
    return roberta_large_model.predict([input_string])[0][0]

y_pred = test_df.text.apply(make_prediction)
test_df[f"pred_{model_name}"] = y_pred

def testing(test_df, model_name, y_pred_column, figure=False):
    """
    This function takes the test dataset and applies the trained model on it to infer predictions.
    It also prints and saves a confusion matrix, calculates the F1 scores and saves the results in a list of results.

    Args:
    - test_df (pandas DataFrame)
    - y_pred_column: column with predictions
    - test_name
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

    # Get the true labels
    y_true = test_df["IPTC_true"].to_list()
    LABELS = list(test_df["IPTC_true"].unique())
    y_pred = test_df[y_pred_column].to_list()

    # Calculate the scores
    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro")
    micro = f1_score(y_true, y_pred, labels=LABELS,  average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Macro f1: {macro:0.3}, Micro f1: {micro:0.3}, Accuracy: {accuracy:0.3}")
    
    print(classification_report(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Get lang specific scores acuracy
    lang_results = {}

    for lang in list(test_df["lang"].unique()):
        current_test_df = test_df[test_df["lang"] == lang]
        current_y_true = current_test_df["IPTC_true"].to_list()
        current_labels = list(current_test_df["IPTC_true"].unique())
        current_y_pred = current_test_df[y_pred_column].to_list()
        cur_macro = f1_score(current_y_true, current_y_pred, labels=current_labels, average="macro")
        cur_micro = f1_score(current_y_true, current_y_pred, labels=current_labels, average="micro")
        cur_accuracy = accuracy_score(current_y_true, current_y_pred)
        lang_results[lang] = {"micro-F1": cur_micro, "macro-F1": cur_macro, "accuracy": cur_accuracy}

    results_acc_df = pd.DataFrame(lang_results)
    print(results_acc_df.transpose().to_markdown())
    
    if figure == True:
        # Plot the confusion matrix:
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        plt.figure(figsize=(9, 9))
        plt.imshow(cm, cmap="Oranges")
        for (i, j), z in np.ndenumerate(cm):
            plt.text(j, i, '{:d}'.format(z), ha='center', va='center')
        classNames = LABELS
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=90)
        plt.yticks(tick_marks, classNames)
        plt.title(f"{model_name}")

        plt.tight_layout()
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        #fig1.savefig(f"Confusion-matrix-{model_name}.png",dpi=100)
    
    return {"micro-F1": micro, "macro-F1": macro, "accuracy": accuracy, "lang_results": lang_results, "report": report}

current_results = testing(test_df, model_name, f"pred_{model_name}")

previous_results[model_name] = current_results

# Save the test_df
test_df.to_json("MaCoCu-manually-annotated-test-set-with-preds.json", orient="records", lines=True)

# Save the result dictionary
with open("results/BERT-model-experiments-results.json", "w") as results_file:
	json.dump(previous_results, results_file)

# Show the results in a table
results_df = pd.DataFrame(previous_results).transpose().drop(columns=["lang_results", "report"])
results_df.reset_index(names="model", inplace=True)
print(results_df.sort_values("macro-F1", ascending=False).to_markdown(index=False))

print("Evaluation completed.")