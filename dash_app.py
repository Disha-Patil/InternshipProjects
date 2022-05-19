"""
In this Dash app, the user can input text in the text box and click on the button 
‘Generate Rank’ to output the table that contains the top three product categories 
predicted by the DistilBERT model.
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
#5)regex   2021.8.3
#6)folium  0.2.1 [folium 0.2.1 is a requirement for the gensim library 
                 #please check https://github.com/python-visualization/folium]
#7)pattern 3.6.0 [Please check https://github.com/clips/pattern]
#8)gensim  4.2.0 [Please check https://radimrehurek.com/gensim/]
#9)dash   2.1.0   [Please check https://dash.plotly.com/installation] 
#10)numpy  1.20.3  

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
import string
import re
import pattern
from gensim.utils import tokenize
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
                # Some basic helper functions to clean text.
                def remove_URL(text):
                    '''
                    text: input text for which the url has to be removed
                    the function remove_URL removes the URL from the text
                    '''
                    url = re.compile(r'https?://\S+|www\.\S+')
                    return url.sub(r'', text)

                def remove_emoji(text):
                    '''
                    text: input text for which the emoji has to be removed
                    the function remove_emoji removes the emoji from the text
                    '''
                    emoji_pattern = re.compile(
                        '['
                        u'\U0001F600-\U0001F64F'  # emoticons
                        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
                        u'\U0001F680-\U0001F6FF'  # transport & map symbols
                        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                        u'\U00002702-\U000027B0'
                        u'\U000024C2-\U0001F251'
                        ']+',
                        flags=re.UNICODE)
                    return emoji_pattern.sub(r'', text)

                def remove_html(text):
                    '''
                    text: input text for which the html has to be removed
                    the function remove_html removes the html from the text
                    '''
                    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
                    return re.sub(html, '', text)

                def remove_non_ascii(text):
                    '''
                    text: input text for which Non ASCII Characters has to be removed 
                    the function remove_non_ascii removes the Non ASCII Characters from the text
                    '''
                    non_ascii_removed = text.encode('ascii', 'ignore').decode('ascii')
                    return non_ascii_removed   

                def remove_character_X_and_x(text):
                    '''
                    text: input text for which more than one occurrence of character X or x has to be removed
                    the function remove_character_X_and_x removes more than one occurrence of character X  or x from the text
                    '''
                    remove_X = re.sub(r'XX+','',text)
                    remove_X_space = re.sub(r'X[/s]X','',remove_X )   
                    remove_x = re.sub(r'xx+','',remove_X_space)
                    remove_x_space = re.sub(r'x[/s]x','',remove_x)
                    return remove_x_space

                def remove_punct(text):
                    '''
                    text: input text for which punctuations has to be removed 
                    the function remove_punct removes the punctuations !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~  from the text
                    '''
                    table = str.maketrans('', '', string.punctuation)
                    return text.translate(table)

                def remove_extra_white_spaces(text):
                    '''
                    text: input text for which the characters [\t\n\r\f\v] has to be replaced by a single space character
                    the function remove_extra_white_spaces replaces the characters [\t\n\r\f\v] by a single space character
                    '''
                    #the characters [\t\n\r\f\v] are replaced by a single space character
                    remove_extra_space_characters = re.sub('/s',' ',text)
                    #replace more than one occurrence of a white space character by an occurrence
                    remove_extra_space = re.sub(' +',' ',remove_extra_space_characters)
                    return remove_extra_space

                def remove_consecutive_repeated_substrings(text):
                    '''
                    text: input text for which the consecutive repeated substring is to be removed
                    the function remove_consecutive_repeated_substrings removes the consecutive repeated substring in text
                    '''
                    while re.search(r'\b(.+)(\s+\1\b)+', text):
                      text = re.sub(r'\b(.+)(\s+\1\b)+', r'\1', text)
                    return text

                # Applying helper functions

                #remove URL from the text
                text_clean = remove_URL(text)
                #remove emoji from text_clean
                text_clean = remove_emoji(text_clean)
                #remove html from text_clean
                text_clean =  remove_html(text_clean)
                #remove Non ASCII Characters from text_clean
                text_clean = remove_non_ascii(text_clean)  
                #remove more than one occurrence of character X and x from text_clean
                text_clean =  remove_character_X_and_x(text_clean)
                #remove punctuations from text_clean
                text_clean = remove_punct(text_clean)

                # Tokenizing 'text_clean' using tokenize function from gensim.utils 
                tokenized_data = list(tokenize(text_clean))

                # Lower casing 'tokenized_data'
                lower_text = str(' '.join([word.lower() for word in tokenized_data]))

                #Remove extra white spaces of lower_text
                extra_space_removed = remove_extra_white_spaces(lower_text)

                #Remove consecutive repeated substrings of extra_space_removed
                consecutive_repeated_substrings_removed = remove_consecutive_repeated_substrings(extra_space_removed)

                #Remove the quotes at the end of consecutive_repeated_substrings_removed
                preprocessed_text =  consecutive_repeated_substrings_removed.strip()
                return preprocessed_text

            #########################################################################
            #Function to create DistilBERT pipeline
            #########################################################################
            def get_rank_dataframe(preprocessed_text):
                """
                preprocessed_text: is the text obtained after pre-processing
                the function get_rank_dataframe returns the dataframe with top3 ranks
                """
                ############################################################################################################
                #Check if GPU is Available or not
                ############################################################################################################
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                ############################################################################################################
                #Write the appropriate paths to retrieve the data 
                ############################################################################################################
                user_path = '/Users/syamalabalasubramanian/Desktop/Github/Complaints_data/EY_INTERNSHIP/With_stopwords/Epochs_5/'
                best_model_path = user_path + "best_model.pt"
                label2id_df_path = user_path + "label2id_df.csv"
                ############################################################################################################
                #Create a label2id dictionary to map label and index
                ############################################################################################################
                labels2id_df = pd.read_csv(label2id_df_path)
                labels = labels2id_df.columns.to_list()
                ids = [int(i) for i in labels2id_df.loc[0,:]]
                id2label = {A: B for A, B in zip(ids, labels)} 
                label2id = {A: B for A, B in zip(labels,ids)} 
                labels = list(labels)
                ############################################################################################################
                #Load the tokenizer
                ############################################################################################################
                tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased',num_labels=len(labels))
                logging.set_verbosity_error()
                ############################################################################################################
                #Load the best model
                ############################################################################################################
                model_best = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                       num_labels=len(labels),
                                                                       id2label=id2label,
                                                                       label2id=label2id)


                model_best.to(device)
                ############################################################################################################
                #SET THE PARAMETERS
                fixed_learning_rate = 1e-5
                fixed_batch_size = 32
                ############################################################################################################
                #LOAD THE OPTIMZER
                #Optimizer
                optimizer_learning_rate = 1e-5
                optimizer_adam_epsilon = 1e-8

                no_decay = ['bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in model_best.named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.2},
                    {'params': [p for n, p in model_best.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.0}
                ]

                optimizer_best = AdamW(optimizer_grouped_parameters, lr = optimizer_learning_rate, eps=optimizer_adam_epsilon)
                ############################################################################################################
                # The function load_chkp is created for loading model
                def load_ckp(checkpoint_fpath, model, optimizer):
                  """
                  checkpoint_path: path to save checkpoint
                  model: model that we want to load checkpoint parameters into       
                  optimizer: optimizer we defined in previous training
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
                ############################################################################################################
                #Get inference from the DistilBERT pipline
                ############################################################################################################  
                #set the device_value
                if device == 'cuda:0':
                    device_value = 0
                else:
                    device_value =-1

                 #Create the DistilBERT pipeline
                MAX_LENGTH = model_best.config.max_position_embeddings

                distilbert_nlp = pipeline(task="sentiment-analysis", return_all_scores = True, model=model_best, tokenizer=tokenizer,
                                          device= device_value, max_length = MAX_LENGTH , truncation=True)

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
    app.run_server(port=1080)

