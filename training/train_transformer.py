"""
Different types of transformers to pre-train here.

"""
import argparse
import pandas as pd
from transformers import DistilBertTokenizerFast, Trainer, TrainingArguments, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score,precision_recall_fscore_support
from torch.nn.functional import softmax
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformer_utils import customDataset









def main():
    # we should add in the training arguments as positional arguments for Trainer class in here and specify the default
    # arguments as in Trainer class.

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default = "./results", help = "directory to store results")
    parser.add_argument("--num_train_epochs", default = 10, help = "number of training epochs")
    parser.add_argument("--per_device_train_batch_size", default = 16, help = "batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", default=64, help ="batch size for evaluation")
    parser.add_argument("--warmup_steps", default = 50, help = "number of warmup steps for learning rate scheduler")
    parser.add_argument("-- weight_decay", default=0.01,help ="strength of weight decay")
    parser.add_argument("--logging_dir", default ="./logs",help = "directory for storing logs")
    parser.add_argument("logging_steps",default = 20, help = "steps when logging")
    parser.add_argument("--no_cuda", default = True, help ="to use gpu or not")

    # args is namespace object with the strings to parse from sys.argv.
    args = parser.parse_args()








if __name__ == "__main__":
    main()
