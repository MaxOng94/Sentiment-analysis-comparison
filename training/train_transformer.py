"""
Different types of transformers to pre-train here.

"""
import argparse








def train():
    # we should add in the training arguments as positional arguments for Trainer class in here and specify the default
    # arguments as in Trainer class.

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default = "./results", help = "directory to store results")

    # args is namespace object with the strings to parse from sys.argv.
    args = parser.parse_args()
    #print(args.square **2)








if __name__ == "__main__":
    train()
