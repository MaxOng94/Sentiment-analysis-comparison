from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, Trainer, TrainingArguments, DistilBertForSequenceClassification
import stanza
from sklearn.metrics import precision_score, f1_score
from training.transformer_utils import customDataset
from tqdm import tqdm
from typing import TypedDict

#========================
# set dataset labels as variables to adapt to other datasets easily
LABEL_COL = "class"
TEXT_COL = "comment"
#==============================
class base:
    """
    First time creating a base class. This base class will contain methods that will be inherited by all of the
    class used throughout the benchmarking.
    """
    def __init__(self) -> None:
        pass

    def read_data(self, fname:str, lower_case: bool=False) ->pd.DataFrame:
        """
        This function will read the textfiles.

        fname will be out of new_train_data.csv, unlabeled_data.csv and test_data.txt

        """
        try:
            df = pd.read_csv(fname, encoding = "UTF-8", usecols = ["class","comment"])
            df[LABEL_COL]= df[LABEL_COL].replace({"negative":0, "neutral":1, "positive":2})
            if lower_case:
                df[TEXT_COL]= df[TEXT_COL].str.lower()

            return df
        except (FileNotFoundError,PermissionError):

            print("No files found. Check the data directory for files.")

    def split_train_test(self, df:pd.DataFrame) ->pd.DataFrame:

        train_texts, val_texts, train_labels, val_labels = train_test_split(df[TEXT_COL], df[LABEL_COL], test_size=.2,random_state = 42)
        train_df = pd.DataFrame(zip(train_labels,train_texts),columns = [LABEL_COL,TEXT_COL])
        test_df = pd.DataFrame(zip(val_labels,val_texts),columns = [LABEL_COL,TEXT_COL])

        return train_df, test_df





    def accuracy(self, df:pd.DataFrame) -> TypedDict:

        """
        This function will return an accuracy score and a F1 score.

        This function takes in the df that contains both true LABEL_COL and predicted LABEL_COL.

        This means that this function has to be called after predict in other LABEL_COLes

        The predict function in every LABEL_COLifier will have to result in a df that contains both true and predicted LABEL_COL
        """

        # change the dtype of LABEL_COL and pred to make sure they are integers
        df[LABEL_COL]= df[LABEL_COL].astype("int32")

        df["pred"]= df["pred"].astype("int32")


        f1 = f1_score(y_true =df[LABEL_COL] , y_pred =df["pred"] , average = "macro")
        acc = precision_score(y_true =df[LABEL_COL] , y_pred =df["pred"], average = "macro")
        metric_dictionary = {"Macro Precision" : acc, "F1": f1}
        return metric_dictionary



class TextBlobSenti(base):

    def __init__(self,file_path:str = None) -> None:

        super(TextBlobSenti, self).__init__()


    def score(self, df:pd.DataFrame) -> pd.Series: # can we just call the file we read from base class?

        df["score"] = df[TEXT_COL].apply(lambda x: TextBlob(x).sentiment.polarity)

        return df["score"]

    def score_to_pred(self,row):
        if row < 0:
            # negative
            return 0
        elif 0 <=row <=0.2:
            # neutral
            return 1
        else:
            # positive
            return 2


    def predict(self,train_file: str,lower_case: bool=False) -> pd.DataFrame:

        df = self.read_data(train_file, lower_case)

        train_df,test_df = self.split_train_test(df)
        test_df["score"] = self.score(test_df)

        # we will bin this to 5 classes, because the number of classes we have is 5. We can change this accordingly if need be
        test_df["pred"]= test_df["score"].apply(self.score_to_pred)

        test_df = test_df.drop(["score"],axis = 1)

        return test_df
    # we can calculcate the accuracy score by using the df from self.predict




class VaderSenti(base):

    def __init__(self,file_path:str = None) -> None:

        super(VaderSenti, self).__init__()
        vader = SentimentIntensityAnalyzer()
        self.vader = vader

    def score(self, df:pd.DataFrame) -> pd.Series: # can we just call the file we read from base class?

        df["score"] = df[TEXT_COL].apply(lambda x: self.vader.polarity_scores(x)["compound"])

        return df["score"]

    def score_to_pred(self,row):
        if row < 0:
            # negative
            return 0
        elif 0 <=row <=0.47:
            # neutral
            return 1
        else:
            # positive
            return 2



    def predict(self,train_file: str,lower_case: bool=False) -> pd.DataFrame:

        df = self.read_data(train_file, lower_case)

        train_df,test_df = self.split_train_test(df)

        test_df["score"] = self.score(test_df)

        # we will bin this to 5 classes, because the number of classes we have is 5. We can change this accordingly if need be
        test_df["pred"]= test_df["score"].apply(self.score_to_pred)


        test_df = test_df.drop(["score"],axis = 1)

        return test_df


class afinnSenti(base):

    def __init__(self, file_path:str = None) -> None:

        super(afinnSenti, self).__init__()

        affinn = Afinn()
        self.affinn = affinn

    def score(self, df:pd.DataFrame) -> pd.Series: # can we just call the file we read from base class?

        df["score"] = df[TEXT_COL].apply(lambda x: self.affinn.score(x))
        df["len"] = df[TEXT_COL].apply(self.get_length)
        df["adjusted"] = df["score"]/df["len"]

        return df["adjusted"]

    def score_to_pred(self,row):
        if row < 0.1:
            # negative
            return 0
        elif 0.1 <=row <=0.15:
            # neutral
            return 1
        else:
            # positive
            return 2


    def predict(self,train_file: str,lower_case: bool=False) -> pd.DataFrame:

        df = self.read_data(train_file, lower_case)

        train_df,test_df = self.split_train_test(df)

        test_df["score"] = self.score(test_df)

        # we will bin this to 5 classes, because the number of classes we have is 5. We can change this accordingly if need be
        test_df["pred"]= test_df["adjusted"].apply(self.score_to_pred)

        test_df = test_df.drop(["score"],axis = 1)

        return test_df



#==========================
"""training should take place in another script.
This transformer class is merely encoding the t
1) Encodes the test sentiments to integers

2) Tokenize will break sentences into tokens and change them to ids.

# requires you to add in batch_size during predictor.py


"""


batch_size = 16

# because this is uncased, we need to lowercase the data

# this class will follow other classes in the classifier.py file just to predict probability and classify.
# training of the transformer model will be in another script
class distilbertSenti(base):


    def __init__(self,file_path: str) -> None:

        # loading model and tokenizer from own directory.

        super(distilbertSenti, self).__init__()

        tokenizer_config_path = f"{file_path.parent.parent}/tokenizer/tokenizer_config"
        model_config_path = f"{file_path}"

        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_config_path)
            self.model= DistilBertForSequenceClassification.from_pretrained(model_config_path)
        except:
            raise Exception("Requires both tokenizer and model config saved under models directory")


    def score(self, df:pd.DataFrame) -> pd.Series:
        """
        Using a trained model, returns a prediction on new data.
        Unlike other classes that bins the prediction under the predict function, we will classify(softmax) under here as it is more
        natural.

        """
        list_text = df[TEXT_COL].tolist()

        #text input to tokenizer must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).
        encodings = self.tokenizer(list_text, truncation = True, padding=True)

        labels = df[LABEL_COL].tolist()

        # pytorch's dataset class
        dataset = customDataset(encodings,labels)
        dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

        device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

        self.model.to(device)

        # model to eval mode

        self.model.eval()

        # empty list for prediction to be appended into
        pred_list = []
        # list for true labels to be appended into, because we turn shuffle = True in data loader
        true = []

        with torch.no_grad():  # so will not update

            # tqdm for progress bar
            loop = tqdm(enumerate(dataloader), total = len(dataloader),
                           leave = True)

            for _, data in loop:
                input_ids = data["input_ids"].to(device)
                masks = data["attention_mask"].to(device)
                labels = data["labels"].to(device)

                outputs = self.model(input_ids, masks, labels = labels)
                # information about model outputs: https://huggingface.co/transformers/main_classes/output.html

                # sample output:
                # SequenceClassifierOutput(loss=tensor(1.6602, device='cuda:0', grad_fn=<NllLossBackward>),
                # logits=tensor([[-0.0098,  0.0775,  0.0358,  0.0997, -0.0220],
                # [-0.0169,  0.1331,  0.0294,  0.1378,  0.0054]], device='cuda:0',
                # grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)

                logits = outputs["logits"]

                # unpack into pred_scores and pred
                predicted_labels = torch.argmax(logits, dim = 1)

                # appends the prediction into an empty list
                pred_list.extend(predicted_labels.cpu().numpy())
                # appends the true labels into empty list
                true.extend(data["labels"])

        df[LABEL_COL] = true
        df["pred"] = pred_list



        return df["pred"]



    def predict(self,train_file: str,lower_case: bool=False) -> pd.DataFrame:

        df = self.read_data(train_file, lower_case)

        train_df,test_df = self.split_train_test(df)

        test_df["pred"] = self.score(test_df)

        return test_df


#===============================================
"""Use pre-trained model from Stanza.
Not sure about training model using stanza, Stanza does not seem to allow us to do so.
"""

class StanzaSenti(base):

    def __init__(self,file_path:str = None) -> None:

        super(StanzaSenti, self).__init__()
        #https://stanfordnlp.github.io/stanza/sentiment.html
        nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
        self.nlp = nlp

    def series_score(self,series:pd.Series) -> int:

        doc = self.nlp(series)

        for sentence in doc.sentences:

            return sentence.sentiment


    def score(self, df:pd.DataFrame) -> pd.Series: # can we just call the file we read from base class?

        df["score"] = df[TEXT_COL].apply(self.series_score)

        return df["score"]



    def predict(self,train_file: str,lower_case: bool=False) -> pd.DataFrame:

        df = self.read_data(train_file, lower_case)

        train_df,test_df = self.split_train_test(df)


        test_df["score"] = self.score(test_df)

        # for consistency, since accuracy function always takes in pred column
        test_df["pred"] = test_df["score"]

        test_df = test_df.drop(["score"],axis = 1)

        return test_df
    # we can calculcate the accuracy score by using the df from self.predict
