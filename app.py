from flask import Flask, render_template,request   
import pandas as pd
import pickle
import itertools
import os
from os  import getcwd
#import bz2
#import _pickle as cPickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'adithya'


directory = getcwd()

smd=pickle.load(open('smd.pkl','rb'))
pre_df=pickle.load(open('pre_df.pkl','rb'))



content=[]
#read csv, and split on "," the line
for index, row in smd.iterrows():
    content.append(row['product_name'])

content2=list(set(content))[:200]





from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['all_meta'])


#Since we have used the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity Score
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

smd = smd.reset_index()
titles = smd['product_name']
#one d array of data
indices = pd.Series(smd.index, index=smd['product_name'])



from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(max_features=None,
                 strip_accents='unicode',
                 analyzer='word',
                 min_df=10,
                 token_pattern=r'\w{1,}',
                 ngram_range=(1,3),#take the combination of 1-3 different kind of words
                 stop_words='english')#removes all the unnecessary characters like the,in etc.

pre_df['description'] = pre_df['description'].fillna('')

#fitting the description column.
tfv_matrix = tfv.fit_transform(pre_df['description'])
#converting everythinng to sparse matrix.

from sklearn.metrics.pairwise import sigmoid_kernel
sig = sigmoid_kernel(tfv_matrix,tfv_matrix)



def similar_prods(title):

  
    idx = indices[title]    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    product_indices = [i[0] for i in sim_scores]
    return titles.iloc[product_indices]


def product_recommendation(title,sig=sig):    
  
    indx = indices[title]
    sig_scores = list(enumerate(sig[indx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]
    product_indices = [i[0] for i in sig_scores]
    return pre_df['product_name'].iloc[product_indices]



@app.route("/")
@app.route("/home", methods=['GET','POST'])
def home():
    
    if request.method=='POST':
        
        prod_name= request.form['prod']
        
        if prod_name in content:
            prod_detail = similar_prods(prod_name)
            prod_detail=prod_detail.head(10).to_dict()
            
            prod_detail_2 = product_recommendation(prod_name)
            prod_detail_2=prod_detail_2.head(10).to_dict()
            
            res={}
            temp={}
            for key,val in prod_detail.items():
                if(val in prod_detail_2):
                    res[key]=prod_detail[key]
                else:
                    temp[key]=prod_detail[key]
            
            l=10-len(res)
            if(l!=0):
                res.update(dict(itertools.islice(temp.items(),l)))
            
            out1 = dict(list(res.items())[:5])
            out2 = dict(list(res.items())[5:])
            
            return render_template('prod_view.html',prod_name=prod_name,prod1=out1,prod2=out2,exists='y') 
        else:
            return render_template('prod_view.html',prod_name=prod_name,exists='n')
    else:
        
        return render_template('index.html', content=content2)


if __name__ == '__main__':
    app.run(debug=True)
    