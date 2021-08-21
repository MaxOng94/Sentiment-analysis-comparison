"""
Running this script will return metrics for classifier that we specify in the command line

1) All trainable classifiers should be trained and model saved under model/"name"

2) Argparser




"""

from classifiers import TextBlobSenti,VaderSenti,afinnSenti,distilbertSenti,StanzaSenti
import argparse
from typing import Any
from pathlib import Path
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from pandas.plotting import table
import pandas as pd
import matplotlib.pyplot as plt
import timeit


# Path to train and test data
TRAIN_PATH = "data/label_data.csv"
# TEST_PATH = "data/label_data.csv"
LABEL_COL = "class"
TEXT_COL = "comment"

METHODS = {
   #rule-based
   #========================
    "textblob":{
        "class":"TextBlobSenti",
        "model": None},
    "vader":{
        "class":"VaderSenti",
        "model":None},
    "afinn":{
        "class":"afinnSenti",
        "model":None},
    #=======================
    #transformers
    # finetune
    #========================
    "distilbert":{
        "class":"distilbertSenti",
        "model": "distilbert/weight_decay=0.5"},
    #========================
    # pretrained
    #====================
    "stanza":{
        "class": "StanzaSenti",
        "model": None}
    }


# method is from args.method
def get_class(method: str) -> Any:
    "instantiate class using its string name"
    # the global function () contains a dictionary of variables (including functions) defined in global namespace of this script.
    method_class = METHODS.get(method).get("class")
    class_ = globals()[method_class]
    # eg: classifiers.StanzaSenti
    return class_


def plot_confusion_matrix(y_true, y_pred,normalize = True):
    cm= confusion_matrix(y_true = y_true, y_pred =y_pred,normalize = "true")
    display= ConfusionMatrixDisplay(cm)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
    return display.plot()


def metric_plot(df):
    fig, ax = plt.subplots(figsize = (10,2))
    ax.set_frame_on(False)
    ax.xaxis.set_visible(False) # hide x axis
    ax.yaxis.set_visible(False) # hide y axis
    table_obj= table(ax, df, colWidths = [0.3]* len(df.columns))
    table_obj.scale(1.0,1.2)  # scale the table obj , scale(width, height)
    fig.savefig("Plots/df.png", bbox_inches = 'tight') # bbox_inches = tight prevents cut off in png
    print(df)


def make_dirs(path:str)->  None:
    if Path(path).exists() != True:
        # if path does not exists
        p = Path(path)
        p.mkdir(parents = True,exist_ok = True)
        print("Created directory {}".format(path))


# method, test_file, lower_case from args.method, args.test_file, args.lower_case
def run_classifier(method: str ,train_file: str, lower_case:bool = False,model_path:str = None):
    class_ = get_class(method)
    # eg: classifiers.StanzaSenti
    # requires to have file when initializaing
    method_obj = class_(model_path)
    test_df = method_obj.predict(train_file)
    # run timeit here to get the time for 5 iterations
    #time_taken = timeit.timeit("method_obj.predict(test_file)", globals = globals(), number =5)
    metric_dictionary = method_obj.accuracy(test_df)
    #metric_dictionary["computation_time"]=time_taken/5
    return test_df, metric_dictionary


def create_plots(method: str,df: pd.DataFrame) -> None:
    make_dirs('Plots')
    display = plot_confusion_matrix(df[LABEL_COL],df["pred"], normalize = True)
    ax, fig= display.ax_ ,display.figure_
    ax.set_title("Normalized confusion matrix : {}".format(method))
    fig.tight_layout()
    fig.savefig("Plots/{}.png".format(method))
    return None





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type = str, default = TRAIN_PATH, help = "Train data file")
    # parser.add_argument("--test", type = str, default = TEST_PATH, help = "Test data file" )
    parser.add_argument("--method", required = True, type =str, help = "method for generating sentiment",nargs = '+')
    parser.add_argument("--lower", action = "store_true", help = "Flag to convert test data to lower case(for classifiers that benefit from lower-casing)")
    parser.add_argument("--model", type =str, help = "Trained classifier model file or path (str)", default = None)
    args = parser.parse_args()

    lower_case = args.lower
    methods = args.method  # this is a list of method

    list_of_df = [] # empty list to append dfs

    for method in methods:
        if method not in METHODS.keys():
            print("Please choose from the existing available model. {}".format(list(METHODS.keys())))
        else:
            train_file = args.train
            # need to split distilbert vs the rest
            if method != "distilbert":
                test_df, metric_dictionary = run_classifier(method, train_file, lower_case)
                create_plots(method, test_df)
                # taken from here: https://stackoverflow.com/questions/5086430/how-to-pass-parameters-of-a-function-when-using-timeit-timer
                time_taken = timeit.Timer(lambda: run_classifier(method, train_file, lower_case)).timeit(number =1)
                metric_dictionary["computation_time"]=time_taken
                df = pd.DataFrame(data = metric_dictionary, index = [method])
                list_of_df.append(df) # appends dfs to list
            else: # for distilbert
                args_model_path = args.model
                try:
                    models = Path('models')
                    model_path = models / args_model_path
                    test_df, metric_dictionary =run_classifier(method, train_file, lower_case = True,model_path= model_path)
                    create_plots(method,test_df)
                    time_taken = timeit.Timer(lambda: run_classifier(method, train_file, lower_case= True,model_path= model_path)).timeit(number =1)
                    metric_dictionary["computation_time"]=time_taken
                    df = pd.DataFrame(data = metric_dictionary, index = [method])
                    list_of_df.append(df)
                except:
                    print("Please choose from the existing available models: {}".format(METHODS[method].get("model")))

    # some sort of hack to prevent error message from appearing from here.
    try:
        final_df = pd.concat(list_of_df)
        metric_plot(final_df)
    except:
        print("")



if __name__ == "__main__":
    main()
