from flask import Flask, render_template,request   
import pandas as pd
import pickle


app = Flask(__name__)
app.config['SECRET_KEY'] = 'adithya'

smd=pickle.load(open('smd.pkl','rb'))

content=[]

for index, row in smd.iterrows():
    content.append(row['product_name'])

content2=list(set(content))[:250]



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



def similar_prods(title):

  
    idx = indices[title]    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    product_indices = [i[0] for i in sim_scores]
    return titles.iloc[product_indices]


pre_df=pickle.load(open('rating.pkl','rb'))

rslt_df = pre_df[pre_df['overall_rating'] > 1.0]  

details = pd.DataFrame(rslt_df, columns =['product_name', 
                                           'overall_rating']) 

rslt_df = details.sort_values(by = ['overall_rating'],ascending=False).head(10)

rslt_df=dict(rslt_df.values)



@app.route("/")
@app.route("/home", methods=['GET','POST'])
def home():
    
    if request.method=='POST':
        
        prod_name= request.form['prod']
        
        if prod_name in content:
            prod_detail = similar_prods(prod_name)
            prod_detail=prod_detail.head(10).to_dict()
            
            out1 = dict(list(prod_detail.items())[:5])
            out2 = dict(list(prod_detail.items())[5:])
            
            return render_template('prod_view.html',prod_name=prod_name,prod1=out1,prod2=out2,exists='y') 
        else:
            return render_template('prod_view.html',prod_name=prod_name,exists='n')
    else:
        
        return render_template('index.html', content=content2)



@app.route("/rating", methods=['GET','POST'])
def rating():
      
    out1 = dict(list(rslt_df.items())[:5])
    out2 = dict(list(rslt_df.items())[5:])
    return render_template('rating.html',out1=out1,out2=out2)
    

if __name__ == '__main__':
    app.run(debug=True)
    
