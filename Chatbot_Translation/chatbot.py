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

import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification

import torch
import time

dataset = pd.read_csv("/content/drive/MyDrive/Chatbot_Translation/data/FAQ.csv")
    
# load a sentence-transformer model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
# encode queries from knowledge base to create corpus embeddings
corpus_embeddings = model.encode(dataset['Query'].tolist(), convert_to_tensor=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

langModel = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=3,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
langModel.to(device)
langModel.load_state_dict(torch.load('/content/drive/MyDrive/Chatbot_Translation/model/finetuned_BERT_GPU_Final.model', map_location=torch.device(device)))
    

# lets put it all together
def get_query_responses(query, top_k=1):
    '''find the closest `top_k` queries of the corpus for the user query based on cosine similarity'''
    
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
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = re.sub('[0-9\-/!@#$%^&*()<>,.{}:~;?"]', '', text)
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    return tokens 
    
def transliterate(english_text):
    response = urlopen("https://inputtools.google.com/request?%s&itc=ta-t-i0-und&num=13&cp=0&cs=0&ie=utf-8&oe=utf-8" % urllib.parse.urlencode({'text': english_text}))
    output = response.read().decode('utf-8')
    output = StringIO(output)
    t =  json.load(output)
    if t[0] == 'SUCCESS':
        return 0, t[1][0][1][0]
    else:
        return 1, ''
        
    
def translate(query):
    translator = Translator()
    translatedQuery = translator.translate(query).text
    return translatedQuery

def identifyLanguage(word):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)
    inputs = tokenizer(word, return_tensors="pt")
    inputs.to(device)
    with torch.no_grad():
        logits = langModel(**inputs).logits
        predicted_class_id = logits.argmax().item()
    return predicted_class_id
    
 
def transformQuery(tokens):
    engTokens = []
    global engCount
    global tamCount
    engCount = 0
    tamCount = 0
    for token in tokens:
        language = identifyLanguage(token)
        if language == 1:
            engTokens.append(token)
            tamCount+=1
        elif language == 2:
            engToken = transliterate(token)
            engTokens.append(engToken[1])
            tamCount+=1
        else:
            engTokens.append(token)
            engCount+=1      
    sentence = ' '.join(engTokens)
    #print(sentence)
    sentence = translate(sentence)
    #print(sentence)
    return sentence,engCount,tamCount
    
    
def get_response(query):
    start = time.time()
    tokens = text_preprocessing(query)
    englishQuery = transformQuery(tokens)
    response = get_query_responses(englishQuery[0])
    #print(response)
    answer = 'Please visit www.abc.lk or call 0771234567 for more details'
    score = float(response[0]['Score'])
    if (score > 0.4):
      answer = response[0]['Response']

    if (englishQuery[1]<englishQuery[2]):
      translator = Translator()
      answer = translator.translate(answer,dest='ta').text
    end = time.time()
    elapsed = end - start
    #print("Matched Score", score, "Processed Time",elapsed)    
    return answer