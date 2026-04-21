import streamlit as st
import torch 
import pickle
import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

#load model 
with open('vocab.pkl','rb') as f:
   vocab = pickle.load(f)
model  = torch.load('model_analyzer.pth',weights_only=False)
model  = SentimentModel(len(vocab))

#title 
st.title("🎭Sentiment Analyzer")


st.write('Enter your review on the product.I catch you emotion and label your opinion')

#user input 
user_input  = st.text_area(
    "Enter your opinion",
    placeholder="Type your review here...",
    height=120
)
if user_input:
   def tokenizer(text):
      
      return text.split(" ")
   def encode (text):
      
      return [vocab.get(word , 0) for word in text]
   max_seq = 50 # as previous model
   def pad_sequence(seq):
      
      if len(seq)<max_seq:
        seq +=[0]*(max_seq-len(seq))
      return seq[:max_seq]
   
   model.eval()
   text  = tokenizer(user_input)
   encoded_text  = encode(text)
   pad_seq = pad_sequence(encoded_text)
   tensor = torch.tensor([pad_seq], dtype=torch.long)
   output = model(tensor)
   pred = torch.argmax(output, dim=1).item()
   labels = {0: "Negative", 1: "Positive",2: "Neutral"}
   result = labels[pred]
   if result == "Positive":
    st.markdown("<h3 style='color:green;'>😊 Positive</h3>", unsafe_allow_html=True)
   elif result == "Negative":
    st.markdown("<h3 style='color:red;'>😡 Negative</h3>", unsafe_allow_html=True)

   else:
    st.markdown("<h3 style='color:orange;'>😐 Neutral</h3>", unsafe_allow_html=True)
    
