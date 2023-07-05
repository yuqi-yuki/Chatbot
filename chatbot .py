

### **Import the datasets**
"""

# Import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot

# Import the dataset
data_faq = pd.read_csv("BankFAQs.csv")

# Display the number of rows and columns of the dataset
data_faq.shape

# Display the first five rows of the dataset
data_faq.head(5)

# Import the second dataset
data_new_class = pd.read_excel("NewClass.xlsx")
data_new_class.head(5)

# Union the two dataset
data = pd.concat([data_new_class, data_faq], ignore_index=True)

data

"""### **Data preprocessing**"""

# Display some descriptive statistics
data.describe()

# Remove duplicates in the column "Question"
data.drop_duplicates(subset=['Question'], inplace=True)

# Display the number of questions/answers in each class (tag/category)
print(data.Class.value_counts())

# Checking for any null values
print('Null values =', data.isnull().values.any()) # There is no null values in the data (Null values = False)

# Display the type of the variables
data.info()

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

bankdata_corpus[:5]
len(bankdata_corpus)

"""### **Vectorization - Bag of words**"""

# Create a bag-of-words representation of the previous preprocessed text
from sklearn.feature_extraction.text import CountVectorizer

# Only single words (1-gram) are used as features
vect = CountVectorizer(ngram_range=(1,1))

# Create a sparse matrix representation of the bag-of-words counts (document-term matrix)
# Each row of the matrix = a question (pattern) in the preprocessed text data
# Each column = a unique word
tdm = vect.fit_transform(bankdata_corpus)

# Convert the sparse matrix to a Pandas data frame
bag_of_words = pd.DataFrame(tdm.toarray(), columns=vect.get_feature_names_out())
bag_of_words.head(10)

"""### **Intent classification**"""

# Select only the target column (here "Class")
Y = data["Class"]

# Divide the dataset in 2 parts : training set (80%) and test set (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(bag_of_words, Y, test_size=0.2, random_state=42)

# Import necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

# Some basic classification algorithms
models = []
models.append(('LR', LogisticRegression(max_iter=1000))) # Logistic regression
models.append(('LDA', LinearDiscriminantAnalysis())) # Linear discriminant analysis
models.append(('KNN', KNeighborsClassifier())) # K-nearest neighbors
models.append(('CART', DecisionTreeClassifier())) # Decision tree
models.append(('NB', GaussianNB())) # Gaussian Naïve Bayes
models.append(('SVM', SVC())) # Support Vector Machine
models.append(('NN', MLPClassifier())) # Neural Network

# Ensemble models 
models.append(('AB', AdaBoostClassifier())) # Adaptative Boosting
models.append(('GBM', GradientBoostingClassifier())) # Gradient Boosting

# Bagging methods
models.append(('RF', RandomForestClassifier())) # Random forest
models.append(('ET', ExtraTreesClassifier())) # Extra trees

# Test these learning classifier models in terms of accuracy
scoring = 'accuracy' # Accuracy = (TP+TN)/(P+N)

# Test options for classification
num_folds = 10
seed = 2

# Test different models of classification
from sklearn.model_selection import KFold, cross_val_score

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed) # Split the data into 10 equally sized folds
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) # Evaluate the performance of each model using the accuracy metric
    results.append(cv_results)
    names.append(name)
    msg = "%s : %f (%f)" % (name, cv_results.mean(), cv_results.std()) # Display the name of the model, the mean and standard deviation of the cross-validation results
    print(msg)

# Visualise the results and compare algorithms
fig = pyplot.figure()
fig.suptitle("Algorithm comparison")
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
fig.set_size_inches(8,4)
pyplot.show()

"""### **Focus on the best model : Neural network**"""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier

# Prepare neural network model
model = MLPClassifier()
model.fit(X_train, Y_train)

# Estimate accuracy on validation set
predictions = model.predict(X_test)

print(f"Accuracy : {accuracy_score(Y_test, predictions)}")
print("\n")
print(classification_report(Y_test, predictions))

# Confusion matrix
df_cm = pd.DataFrame(confusion_matrix(Y_test, predictions), columns=np.unique(Y_test), index = np.unique(Y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'

print(df_cm)

"""### **Entity extraction (extract keywords to match questions/patterns with the answers)**"""

# Pre processing of the user question

# Enter the user query
input_text = "Can i open a new bank account on the app"

# Set the lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess the user's question
user_question = input_text.lower().strip() # Put in lower case and remove blank spaces
user_question = re.sub(r'[0-9]', '', user_question) # Remove numbers
user_question = re.sub(r'[^\w\s]', '', user_question) # Remove punctuation
user_question = lemmatizer.lemmatize(user_question) # Lemmatization
user_question = " ".join([word for word in word_tokenize(user_question) if word not in stop_words]) 
# Remove stopwords and join words into a single string

print(user_question)
type(user_question)

# Vectorization of the user question
# Split the vector 
user_question_splited=user_question.split(" ")

# Create vector with colnames of bag of word and nan values
user_question_vectorized = pd.DataFrame(columns=bag_of_words.columns, index=[0])

# Go through the bag of word and put 1 if the word match and 0
for i in user_question_splited:
  for j in bag_of_words.columns:
    if i == j:
      user_question_vectorized.loc[0,j] = 1
    else:
      user_question_vectorized.loc[0,j] = 0

print(user_question_vectorized)

# Predict the class of the user query
user_question_class = model.predict(user_question_vectorized)
print(user_question_class)

# Add the class to the dataframe of vectorized questions
bag_of_words_class = bag_of_words
bag_of_words_class['Class'] = data['Class']

# Select the vectors within the same class as the predicted query 
bag_of_words_corresponding_class = bag_of_words_class.loc[bag_of_words_class['Class'] == user_question_class[0]]
print(bag_of_words_corresponding_class)

# Delete the class colunm 
bag_of_words_corresponding_class = bag_of_words_corresponding_class.drop(['Class'], axis=1)
bag_of_words_class = bag_of_words_class.drop(['Class'], axis=1)
bag_of_words = bag_of_words.drop(['Class'], axis=1)

# Get the index of the question with the highest cosine similarity to the user's question
from sklearn.metrics.pairwise import cosine_similarity
# Create an array with all the cosin bewteen the vector query and the questions 
cosine_similarities = []

for i in range(len(bag_of_words_corresponding_class.index)):
  cosine_similarities.append(cosine_similarity(user_question_vectorized, bag_of_words_corresponding_class.iloc[i].values.reshape(1, -1)))

max(cosine_similarities)

# Get the higher cosine
most_similar_question_index_in_same_class_dataframe = np.array(cosine_similarities).argmax()
most_similar_question_index_of_answer = bag_of_words_corresponding_class.iloc[most_similar_question_index_in_same_class_dataframe].name

print(most_similar_question_index_of_answer)

# Get the answer associated with the most similar question
response = data.iloc[most_similar_question_index_of_answer]['Answer']
print(response)

question = data.iloc[most_similar_question_index_of_answer]['Question']
print(question)