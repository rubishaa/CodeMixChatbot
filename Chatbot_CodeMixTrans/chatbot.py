import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator

import urllib
import json
from io import StringIO
from urllib.request import urlopen
import urllib.parse

import re
from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
from yaml.tokens import Token
from nltk.tokenize import word_tokenize
import string
import nltk
import time

import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification



dataset = pd.read_csv("/content/drive/MyDrive/Chatbot_BERT/data/FAQ.csv")
    
# load a sentence-transformer model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
# encode queries from knowledge base to create corpus embeddings
corpus_embeddings = model.encode(dataset['Query'].tolist(), convert_to_tensor=True)

def get_query_responses(query, top_k=3):
    '''find the closest `top_k` queries of the corpus for the user 
    query based on cosine similarity'''
  
    # encode user query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # use cosine-similarity and torch.topk to find the highest `top_k` scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, dataset.shape[0]))
    
    # filter dataframe by list of index
    df = dataset.iloc[top_results[1], :]
    
    # add matched score
    df['Score'] = ["{:.4f}".format(value) for value in top_results[0]]
    # select top_k responses
    responses = df.to_dict('records')
    
    return responses
            
        
def text_preprocessing(text):
   
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
     # Remove punctuations
    text = re.sub('[0-9\-/!@#$%^&*()<>,.{}:~;?"]', '', text)
        
    return text
  
    
def get_response(query):
    start = time.time()
    text = text_preprocessing(query)
    response = get_query_responses(text)
    answer = 'Please visit www.abc.lk or call 0771234567 for more details'
    score = float(response[0]['Score'])
    if (score > 0.3):
      answer = response[0]['Response']
    end = time.time()
    elapsed = end - start
    print(response)
    #print("Matched Score", score, "Processed Time",elapsed)
    return answer,score,elapsed