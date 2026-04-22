# 📈 Product Sentiment Analyzer 

# 🔐 Problem Statement :

Most of business want to know the opinion of customer review.Knowing it nomally or manually reading reviews take lot of time.To avoid this problem i implemented this system.
# 🎯Objective :
The objective is to develop a senttiment analyzer system by using deep learning (LSTM).

The system should be trained by using data set.
# ⚙️Tech Stack :
Deep Learning (LSTM)

Pandas 

Python 

Stream -lit
# APP DEMO :
<img width="1837" height="825" alt="Screenshot 2026-04-22 005740" src="https://github.com/user-attachments/assets/9dcd8326-1560-450b-918e-1b0ff5a8686c" />

#Approach :
I went through the following steps to implement this system end to end .

# Step-1 :Data collection and Data Inspection

pandas.read_csv()
data.info()
# Step-2: Data preprocessing and Text preprocessing 

Checking null values 

Type conversations 

Text Splitting (str.split())

Embeddings creations for the text 

Pad sequences for each text 

# Step -3 : Feature Engineering 
#Feature selection

Tensor array creations for pandas list

Splitting features for testing , training

# Step-4 : Model buidling(LSTM)

Developed sentiment model through classes , LSTM pre build architecture

# Step-5 : Loss and Optimizers 
Usef ClassEntropyClass() , optim optimizer to perform loss handling

# Step-6 : Prediction 
Predicted test data 

Finally model generated labels

# Impact on Business :
-Emotion of the customer 

-Review Labeling 
