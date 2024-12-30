import minsearch
import json

import streamlit as st
import time

from tqdm.auto  import tqdm
from openai import OpenAI

from elasticsearch import Elasticsearch



es_client = Elasticsearch('http://localhost:9200')


client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama')









index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}





prompt_template = """
You're a course teaching assistant. Answer the question based on the context from the FAQ database.
Use only the facts from the CONTEXT when answering the Question.

QUESTION: {question}

CONTEXT: {context}

""".strip()



def minsearch_result(query):

    with open('/Users/adebimpe/Desktop/llm_zoomcamp/02-open-source/frontend/documents.json','rt') as f_in:
        docs_raw = json.load(f_in)

    documents = []


    for course_dict in docs_raw:
        for doc in course_dict['documents']:
            doc['course'] = course_dict['course']
            documents.append(doc)
            

    index = minsearch.Index(
        text_fields=["question","text","section"],
        keyword_fields=['course']
        )

    index.fit(documents)



    boost = { 'question':3.0,'section':0.5}

    search_results = index.search(
            query=query,
            filter_dict={'course':'data-engineering-zoomcamp'},
            boost_dict=boost,
            num_results=5
        )
    
    return search_results



def elastic_search_result(query):


    search_query = {
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^3", "text", "section"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "data-engineering-zoomcamp"
                }
            }
        }
    }
   }


    index_name = "course-questions"

    try:
       es_client.indices.create(index=index_name,body=index_settings)
    except:
       pass
    



    with open('/Users/adebimpe/Desktop/llm_zoomcamp/02-open-source/frontend/documents.json','rt') as f_in:
        docs_raw = json.load(f_in)

    documents = []


    for course_dict in docs_raw:
        for doc in course_dict['documents']:
            doc['course'] = course_dict['course']
            documents.append(doc)

    for doc in tqdm(documents):
        es_client.index(index=index_name, document=doc)

    response = es_client.search(index=index_name,body=search_query)

    result_docs = []


    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs



def create_context(search_results):
    context = ""


    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    return context




def  create_prompt(context,question,prompt_template):


    prompt = prompt_template.format(question=question,context=context).strip()

    return prompt


def get_query_result(prompt):
    response = client.chat.completions.create(model='phi3',messages=[{"role":"user","content":prompt}])
    results = response.choices[0].message.content
    return results



def rag(query,prompt_template,search_engine=None):
    if search_engine == "minsearch":
       search_results = minsearch_result(query)
    else:
       search_results = elastic_search_result(query)

    context = create_context(search_results)
    prompt = create_prompt(context=context,question=query,prompt_template=prompt_template)
    print(prompt)
    
    return get_query_result(prompt)







def main():

    
                                 
    st.title("RAG Function Invocation")

    user_input = st.text_input("Enter your input:")

    search_engine = st.selectbox('search_engine' , ('minsearch','elastic_search'))

    if st.button("Ask"):
        with st.spinner('Processing...'):
            output = rag(user_input,prompt_template,search_engine)
            st.success("Completed!")
            st.write(output)

if __name__ == "__main__":
    main()