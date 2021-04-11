# import specific classifier



# adapted from paper : Neural-Semi-supervised-Learning-for-Text-Classification-Under-Large-Scale-Pretraining
# https://arxiv.org/pdf/2011.08626.pdf

# Define terminologies
# U: unlabeled data
# D: orignally labeled data from before
# U': newly labeled data
# D': selected set from newly labeled data

# Instructions:
# 1) Label unlabeled data using a given model, we will call this U' --> do we want to allow different models her
#   a) Prediction will return score, prediction label and sentence
# 2) Given a threshold value for selecting scores/ select the k highest score labels from the trained model for each class, call this D'
# 3) Combine the originally labeled data with selected set from newly labeled data
#   a) D + D'
# 4) Export the combined data as data for student model


# prediction from teacher model will be saved in the file path data
# newly combined data will also be saved in file path data



# this will just take in the orignal data D and newly labeled data U'
# extract a certain subset of data from U' to create D' --> argparse
# combine and return data for student model


# assume newly labeled data U' in data. Extract it now.
import argparse
import random
import pathlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_labeled_data_path", required= True, type=str, help = "labeled data we have")
    parser.add_argument("--percent_select", required = True, type=float, help = "percentage of data selected from newly labeled data")
    parser.add_argument("--save_path", required = True, type=str, help = "path to save combined data")
    args = parser.parse_args()

    with open(args.original_labeled_data_path, mode = 'r', encoding = 'utf-8') as f:
        train_lines = f.readlines()




if __name__ == '__main__':
    main()
