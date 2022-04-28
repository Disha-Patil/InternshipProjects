#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:02:20 2022

@author: syamalabalasubramanian
"""
##################################
#Requirements
##################################
#packages  #version
#################
#1)torch   1.11.0
#2)transformers  4.11.3 [Please check https://huggingface.co/docs/transformers/installation ]   
#3)datasets  1.18.4 [Please check https://huggingface.co/docs/datasets/installation ] 
#4)pandas  1.3.4
#5)spacy  3.2.1 [Please check https://spacy.io/usage]
#6)spacy-model-en_core_web_sm   3.2.0  [Please check https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.2.0]
#7)regex   2021.8.3 
#8)dash   2.1.0   [Please check https://dash.plotly.com/installation] 
#9)numpy  1.20.3  

##################################
#Import the librarires
##################################
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import logging
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW 
from transformers import pipeline
import pandas as pd
import spacy
import re
import dash
import numpy as np
import time
from dash import html ,Input, Output, State, dcc, dash_table

text = "Sorry, there's nothing in the text input. Please write something."

# Dash app starts here
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='DistilBERT Top 3 Rank'),

    dcc.Textarea(
        id='textarea-input',
        value=text,
        style={'width': '100%', 'height': '45vh'}
    ),
    dcc.Loading([
        html.Button("Generate Rank", id='button'),

    ]),
    html.Hr(),
    html.Div(id="table1",children=[dash_table.DataTable(id='table-editing-simple-output')]),
])

@app.callback(Output('table1','children'),
              [Input("button", "n_clicks")],
              [State("textarea-input", "value")])
def update_output(n_clicks, text):
    if n_clicks is not None:
        ##########
        def get_top3_rank(text):
            
            """
            text: the user-typed input text
            the function get_top3_rank outputs rank_df the dataframe that contains the top 3 ranks
            """
            def text_pre_process(text):
                """
                text: the user-typed input text
                the function text_pre_process outputs change_text that is the pre-processed text
                """
                #########################################################################
                #Create Spacy Doc
               #########################################################################
                spacy_nlp = spacy.load("en_core_web_sm")
                text_doc = spacy_nlp(text)
               #########################################################################
                #Define the strings to mask
                mask_words_list =['XX/XX/XXXX','XX-XX-XXXX', #DATE mm/dd/yyyy mm-dd-yyyy
                                  'XXXX XXXX XXXX XXXX XXXX','XXXX-XXXX-XXXX-XXXX',#CREDIT or PREPASID CARD NUMBER
                                  'XXXX XXXX XXXX XXXX','XXXX XXXX XXXX','XXXX-XXXX-XXXX','XXXX-XXXX','XXXX XXXX',
                                  'XXX-XX-XXXX','XXX-XXX','XX-XXXX',
                                  'XXXXXXXXXXXXXXXXXX','XXXXXXXXXXXXXXXXX', 'XXXXXXXXXXXXXXXX', 'XXXXXXXXXXXXXXX', 'XXXXXXXXXXXXXX',# BANK ACCOUNT NUMBER
                                  'XXXXXXXXXXXXX', 'XXXXXXXXXXXX', 'XXXXXXXXXXX',                                                   # RANGES FROM 12 TO 18 DIGITS
                                  'XXXXXXXXXX','XXXXXXXXX'          #ROUTING NUMBER IS 9 DIGIT
                                  'XXXX','XXX','XX']
                #########################################################################
                #Pre-process the Text
                #########################################################################
                def change_details(word):
                    """
                    word: is the word contained in the text_doc
                    the function change_details outputs string depending on the type of the word
                    """
                    if word.like_email or word.like_url:
                        return '<MASK>'
                    elif any(mask_word in word.text for mask_word in mask_words_list):
                        return '<MASK>'
                    elif word.is_stop:
                        return ''
                    elif (len(re.findall('\.',word.text)) < 1) :
                        if word.is_punct:
                            return ''
                    return word.text

                # Function where each token of spacy doc is passed through change_details()
                def change_text(doc):
                    """
                    doc: is the spacy document of the corresponding text
                    the function change_text returns the preprocessed string
                    """
                    bert_threshold = 230
                    # Passing each token through change_details() function.
                    new_tokens = map(change_details,doc)
                    new_text = str(' '.join(new_tokens))
                    #Replace more than one white space in the string with one white space
                    new_text = re.sub(' +', ' ',new_text)
                    new_text = new_text.replace(' .', '.')
                    new_text = new_text.replace('\n', '')
                    new_text_words  = new_text.split()
                    if len(new_text_words) < bert_threshold:
                        return new_text
                    else :

                        truncated_text_words  = new_text_words[:bert_threshold]
                        truncated_text = str( " ".join(truncated_text_words))
                        return truncated_text

                changed_text = change_text(text_doc)
                return changed_text
            #########################################################################
            #Function to create DistilBERT pipeline
            #########################################################################
            def get_rank_dataframe(preprocessed_text):
                """
                preprocessed_text: is the text obtained after pre-processing
                the function get_rank_dataframe returns the dataframe with top3 ranks
                """
                #########################################################################
                #Check if GPU is Available or not
                #########################################################################
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                #########################################################################
                #Write the appropriate paths to retrieve the data 
                #########################################################################
                user_path = "/Users/syamalabalasubramanian/Desktop/Github/Complaints_data/Sample_Data/Experiment3/Submission/Experiment3_Dash_APP/"
                best_model_path = user_path + "best_model.pt"
                label2id_df_path = user_path + "label2id_df.csv"
                #########################################################################
                #Create a label2id dictionary to map label and index
                #########################################################################
                labels2id_df = pd.read_csv(label2id_df_path)
                labels = labels2id_df.columns.to_list()
                ids = [int(i) for i in labels2id_df.loc[0,:]]
                id2label = {A: B for A, B in zip(ids, labels)} 
                label2id = {A: B for A, B in zip(labels,ids)} 
                labels = list(labels)
                #########################################################################
                #Load the tokenizer
                #########################################################################
                tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased',num_labels=len(labels))
                #########################################################################
                #Load the best model
                ##########################################################################
                logging.set_verbosity_error()

                model_best = AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased",
                                                                   num_labels=len(labels),
                                                                   id2label=id2label,
                                                                   label2id=label2id)


                model_best.to(device)
                #SET THE PARAMETERS
                fixed_learning_rate = 1e-5
                fixed_batch_size = 16
                #LOAD THE OPTIMZER
                optimizer_best = AdamW(model_best.parameters(),lr = fixed_learning_rate)
                #load_chkp is created for loading model.
                def load_ckp(checkpoint_fpath, model, optimizer):
                    """
                    checkpoint_path: path to save checkpoint
                    model: model that we want to load checkpoint parameters into       
                    optimizer: optimizer we defined in previous training
                    the function load_ckp returns the model, optimizer, checkpoint, valid_loss_min from the checkpoint_path
                    """
                    # load check point
                    checkpoint = torch.load(checkpoint_fpath,map_location=torch.device('cpu'))
                    # initialize state_dict from checkpoint to model
                    model.load_state_dict(checkpoint['state_dict'])
                    # initialize optimizer from checkpoint to optimizer
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    # initialize valid_loss_min from checkpoint to valid_loss_min
                    valid_loss_min = checkpoint['valid_loss_min']
                    # return model, optimizer, epoch value, min validation loss 
                    return model, optimizer, checkpoint['epoch'], valid_loss_min

                # load the saved checkpoint
                model_best, optimizer_best, end_epoch, valid_loss_min = load_ckp(best_model_path, model_best, optimizer_best)

                #########################################################################
                #Get inference from the DistilBERT pipline
                ##########################################################################   
                #set the device_value
                if device == 'cuda:0':
                    device_value = 0
                else:
                    device_value =-1

                #Create the DistilBERT pipeline
                distilbert_nlp = pipeline(task="sentiment-analysis", return_all_scores = True, model=model_best, tokenizer=tokenizer,device= device_value)
                #input the pre-processed text
                results = distilbert_nlp(preprocessed_text)
                #########################################################################
                #Return the result in a data frame
                ##########################################################################   
                pd.set_option('display.max_colwidth', None)
                pd.set_option('precision', 4)
                #rank the inferences by the prediction score 
                result_df = pd.DataFrame(results[0]).sort_values(by='score', ascending=False)
                result_df.index = range(1,len(labels)+1)
                #truncate to store the top 3 ranks only
                result_rank = result_df.loc[:3,:]
                result_rank = result_rank.drop(['score'],axis=1).reset_index().rename(columns={'index':'rank'}) 
                result_inf = result_rank
                return result_inf
            changed_text= text_pre_process(text)
            rank_df = get_rank_dataframe(changed_text)
            return rank_df
        #call the function get_top3_rank using the input 'text' to return the dataframe with top 3 rank
        result_inf = get_top3_rank(text)
        return html.Div([dash_table.DataTable( id='table-output',
                data=result_inf.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in result_inf.columns],
                ),
               html.Hr()
        ])
    
if __name__ == '__main__':
    app.run_server(port=1090)







