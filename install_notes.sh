#Need to be signed into kaggle to get the data, at which point it needs to be downloaded from:
#https://www.kaggle.com/competitions/nlp-getting-started/data


#Code to setup a virtual environment
#TODO
virtualenv dis_tweets
source dis_tweets/bin/activate
pip install -r requirements.txt

#Create a folder to store the data from both kaggle and GloVe vectors
mkdir raw_data

#Code to deal with the GloVe vectors; which includes a long download (1.5 GB ish)
wget -O glove.twitter.27B.zip --show-progress https://nlp.stanford.edu/data/glove.twitter.27B.zip 
mv glove.twitter.27B.zip raw_data/
cd raw_data
unzip glove.twitter.27B.zip
cd ..

#

