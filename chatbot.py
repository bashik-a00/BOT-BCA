import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Read the chatbot data
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Split the text into paragraphs
para_tokens = raw.split('\n\n')  # Assumes paragraphs are separated by double newlines

lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response
def response(user_response):
    robo_response = ''
    para_tokens.append(user_response)

    tfidfvec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', token_pattern=None)
    tfidf = tfidfvec.fit_transform(para_tokens)

    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][:-1][::-1]  # Get indices of paragraphs in descending order of similarity

    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        robo_response += "I am sorry! I don't understand you"
    else:
        for i in idx:
            if vals[-1][i] > 0.1:  # Threshold to consider relevant paragraphs
                robo_response += para_tokens[i] + '\n\n'
            if len(robo_response) > 0:
                break

    para_tokens.pop(-1)  # Remove the user response added at the end
    return robo_response.strip()

flag = True
print("BOT-BCA: My name is BOT-BCA. I will answer your queries about BCA. If you want to exit, type Bye!")

while flag:
    user_response = input().lower()

    if user_response != 'bye':
        if user_response in ['thanks', 'thank you']:
            flag = False
            print("BOT-BCA: You are welcome..")
        else:
            if greeting(user_response) is not None:
                print("BOT-BCA: " + greeting(user_response))
            else:
                print("BOT-BCA: ", end="")
                print(response(user_response))
    else:
        flag = False
        print("BOT-BCA: Bye! Take care..")
