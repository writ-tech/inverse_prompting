from transformers import pipeline, set_seed
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

def main():
    
    st.title("Awesome Streamlit for ML")

    prompt = st.text_input('Movie title')
    

    generator = pipeline('text-generation', model='gpt2')


    f=generator("Question: "+prompt+" Answer:", max_length=50,  num_beams=5,
                no_repeat_ngram_size=2,    num_return_sequences=4,    early_stopping=True, 
    )
    m=[]
    f1=generator("Question: "+prompt+" Answer:", max_length=50,  
                no_repeat_ngram_size=2,    early_stopping=True, 
    )
    len1=len("Question: "+prompt+" Answer:")
    ans=f1[0]['generated_text']
    ans1=ans[len1:]
    len1=len("Question: "+prompt+" Answer:")
    for i in f:
      c=i['generated_text']
      c=c[len1:]
      m.append(c)
    ar=[]
    for i in m:
      e=generator("Answer:"+i+ "answers the question:", max_length=60,  
                num_beams=5,
                no_repeat_ngram_size=2, 
                num_return_sequences=5, 
                     early_stopping=True, 
       )
      
      k=len("Answer:"+i+ "answers the question: ")
      p=[]
      for j in e:


        p.append(j['generated_text'][k:])
        
      ar.append(p)

      
    
    model = SentenceTransformer('bert-base-nli-mean-tokens')
  
    sen_embeddings = model.encode(prompt)
 
    em=[]
    for i in ar:
      em.append(model.encode(i[0].partition('?')[0]))
     


    s=cosine_similarity(
          [sen_embeddings],em
        
    )

    print(s)
    #st.write(s)
    ind=np.argmax(s)
    st.write('With inverse prompting:',m[ind])
    st.write('Without inverse prompting:',m[0])



	
if  __name__ == '__main__':
    
    
    main()
    
