"""
Different types of transformers to pre-train here.

"""
import argparse
import pandas as pd
from transformers import DistilBertTokenizerFast, Trainer, TrainingArguments, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score,precision_recall_fscore_support
from torch.nn.functional import softmax
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformer_utils import customDataset, read_data


train_file = "../data/label_data.csv"
test_file = "../data/label_data.csv"
LABEL_COL = "class"
TEXT_COL = "comment"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall}




def main():
    # we should add in the training arguments as positional arguments for Trainer class in here and specify the default
    # arguments as in Trainer class.

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default = "./results", help = "directory to store results")
    parser.add_argument("--num_train_epochs", default = 10, help = "number of training epochs")
    parser.add_argument("--per_device_train_batch_size", default = 16, help = "batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", default=64, help ="batch size for evaluation")
    parser.add_argument("--warmup_steps", default = 50, help = "number of warmup steps for learning rate scheduler")
    parser.add_argument("--weight_decay", default=0.01,help ="strength of weight decay")
    parser.add_argument("--logging_dir", default ="./logs",help = "directory for storing logs")
    parser.add_argument("--logging_steps",default = 20, help = "steps when logging")
    parser.add_argument("--no_cuda", default = True, help ="to use gpu or not")

    # args is namespace object with the strings to parse from sys.argv.
    args = parser.parse_args()

    train_df = read_data(train_file, LABEL_COL,TEXT_COL,lower_case = True)
    labels = train_df[LABEL_COL]

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_df[TEXT_COL], train_df[LABEL_COL], test_size=.2,random_state = 42)

    train_texts = train_texts.tolist()
    val_texts = val_texts.tolist()
    train_labels= train_labels.tolist()
    val_labels= val_labels.tolist()

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels = 3)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = customDataset(train_encodings, train_labels)
    val_dataset = customDataset(val_encodings, val_labels)

    training_args = TrainingArguments(output_dir = args.output_dir,
                                        num_train_epochs = args.num_train_epochs,
                                        per_device_train_batch_size= args.per_device_train_batch_size,
                                        per_device_eval_batch_size= args.per_device_eval_batch_size,
                                        warmup_steps= args.warmup_steps,
                                        weight_decay= args.weight_decay,
                                        logging_dir = args.logging_dir,
                                        logging_steps = args.logging_steps,
                                        no_cuda = args.no_cuda)


    trainer = Trainer(model = model,
                       args = training_args,
                       train_dataset = train_dataset,
                       eval_dataset = val_dataset,
                        compute_metrics = compute_metrics)



    trainer.save_model("../models/distilbert/model_config")

    tokenizer.save_pretrained('../models/distilbert/tokenizer_config')






if __name__ == "__main__":
    main()
