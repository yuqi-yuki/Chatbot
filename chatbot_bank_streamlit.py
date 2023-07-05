

#----------------
#model
#----------

#### Import packages ####

# Import necessary package
import pandas as pd
import numpy as np
import streamlit as st
#from streamlit_chat import message
from typing import Literal

# Import packages for data preprocessing
import random  
import string
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.metrics.pairwise import cosine_similarity



#### Import datasets ####

# Import the dataset
data_faq = pd.read_csv("/Users/syq/Downloads/BankFAQs.csv")

# Import the second dataset
data_new_class = pd.read_excel("/Users/syq/Downloads/NewClass.xlsx")

# Union the two dataset
data = pd.concat([data_new_class, data_faq], ignore_index=True)



#### Data preprocessing ####

# Remove duplicates in the column "Question"
data.drop_duplicates(subset=['Question'], inplace=True)

# Removing special characters in non latin language
usable_question = data['Question'].str.encode('ascii', 'ignore').str.decode('ascii')

# Corpus creation and data cleaning
bankdata_corpus = [text.lower() for text in usable_question] # Put in lower case
bankdata_corpus = [re.sub(r'[0-9]', '', text) for text in bankdata_corpus] # Remove numbers
bankdata_corpus = [text.strip() for text in bankdata_corpus] # Remove blank spaces
bankdata_corpus = [re.sub(r'[^\w\s]', '', text) for text in bankdata_corpus] # Remove punctuation

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Join the words in each sublist of "bankdata_corpus" into a single string using a space as a separator because "bankdata_corpus" variable is a list of lists and not a list
bankdata_corpus = [" ".join([lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words]) for text in bankdata_corpus]



#### Vectorization - Bag of words ####

# Create a bag-of-words representation of the previous preprocessed text
# Only single words (1-gram) are used as features
vect = CountVectorizer(ngram_range=(1,1))

# Create a sparse matrix representation of the bag-of-words counts (document-term matrix)
# Each row of the matrix = a question (pattern) in the preprocessed text data
# Each column = a unique word
tdm = vect.fit_transform(bankdata_corpus)

# Convert the sparse matrix to a Pandas data frame
bag_of_words = pd.DataFrame(tdm.toarray(), columns=vect.get_feature_names_out())



#### Intent classification ####

# Select only the target column (here "Class")
Y = data["Class"]

# Divide the dataset in 2 parts : training set (80%) and test set (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(bag_of_words, Y, test_size=0.2, random_state=42)



#### Focus on the best model : Neural network ####

# Prepare neural network model
model = MLPClassifier()
model.fit(X_train, Y_train)

# Estimate accuracy on validation set
predictions = model.predict(X_test)



#### Entity extraction (extract keywords to match questions/patterns with the answers) ####

# Set the lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define function to preprocess user's question
def preprocess_question(question):
    question = question.lower().strip()
    question = re.sub(r'[0-9]', '', question) # Remove numbers
    question = re.sub(r'[^\w\s]', '', question) # Remove punctuation
    question = lemmatizer.lemmatize(question)
    question = " ".join([word for word in word_tokenize(question) if word not in stop_words])
    return question

# Define function to vectorize user's question
def vectorize_question(question):
    question_splited = question.split(" ")
    question_vectorized = pd.DataFrame(columns=bag_of_words.columns, index=[0])
    for i in question_splited:
        for j in bag_of_words.columns:
            if i == j:
                question_vectorized.loc[0,j] = 1
            else:
                question_vectorized.loc[0,j] = 0
    return question_vectorized

# Define function to get the answer for the user's question
def get_answer(question):
    global bag_of_words
    user_question = preprocess_question(question)
    user_question_vectorized = vectorize_question(user_question)
    user_question_class = model.predict(user_question_vectorized)
    bag_of_words_class = bag_of_words
    bag_of_words_class['Class'] = data['Class']
    bag_of_words_corresponding_class = bag_of_words_class.loc[bag_of_words_class['Class'] == user_question_class[0]]
    bag_of_words_corresponding_class = bag_of_words_corresponding_class.drop(['Class'], axis=1)
    bag_of_words_class = bag_of_words_class.drop(['Class'], axis=1)
    bag_of_words = bag_of_words.drop(['Class'], axis=1)
    cosine_similarities = []

    for i in range(len(bag_of_words_corresponding_class.index)):
        cosine_similarities.append(cosine_similarity(user_question_vectorized, bag_of_words_corresponding_class.iloc[i].values.reshape(1, -1)))

    most_similar_question_index_in_same_class_dataframe = np.array(cosine_similarities).argmax()
    most_similar_question_index_of_answer = bag_of_words_corresponding_class.iloc[most_similar_question_index_in_same_class_dataframe].name
    response = data.iloc[most_similar_question_index_of_answer]['Answer']
    return response

#-------------------
#web
#-----------

import streamlit as st
from streamlit_chat import message
from typing import Literal



# set characters
APP_TITLE = 'Chatbot for bank'

message_history=[]
input_history=[]
avatar_style = 'big-ears'
avatar_url = f"https://api.dicebear.com/5.x/{avatar_style}.svg"
avatar_html = f'<img src="{avatar_url}" alt="avatar">'
logo = ["https://chaire-sirius.eu/i/13.jpeg?fm=jpg&w=350&h=350&fit=crop&s=6b86c57772e2dcbee30e094d6c8e5234",
        "https://media.istockphoto.com/id/1010001882/fr/vectoriel/%C3%B0-%C3%B0%C2%B5%C3%B1-%C3%B0-%C3%B1-%C3%B1.jpg?s=612x612&w=0&k=20&c=1eSGWj2ckLNZrjRenFfhTNIN3GkHdqeZ365nlM0gvsA="]

def main():
    st.image(logo, width=200)
    # set app title
    st.title(APP_TITLE)
    
    message('Welcome to the Chatbot. I am still learning,please be patient', is_user=False)
    placeholder = st.empty()
    


    # set an input box
    input_ = st.text_input("Enter your question")   
    input_history.append(input_)


    response = get_answer(input_)
    message_history.append(response)   
    
    with placeholder.container():
        for msg in input_history:
            message(msg, 
            is_user=True, 
            avatar_style=avatar_style, # change this for different user icon
             seed=123 # or the seed for different user icons
             ) 
        
        for x in message_history:
            message(x,
                    is_user = False)
        
    # with placeholder.container():
    #     for msg in (message_history,input_history):
    #         if msg in input_history:
    #             message(msg, 
    #             is_user=True, 
    #             avatar_style=avatar_style, # change this for different user icon
    #             seed=123 # or the seed for different user icons
    #         )   
    #         if msg in message_history:
    #             message(msg,
    #             is_user=False, # or the seed for different user icons
    #         )

     # st.write("")
     #            st.write("")    
     #            st.write("")
     #            st.write("")
     #            st.write("")
     #            st.write("")    
     #            st.write("")
     #            st.write("")
                

if __name__ == '__main__':
                main()