import json
import pandas as pd
import minsearch
from openai import OpenAI
from template import prompt_template , entry_template , eval_template
import time


open_client = OpenAI(api_key="")

data_path = 'data/data.csv'

def load_documents():
      
    data_df = pd.read_csv(data_path)

    documents = data_df.to_dict(orient='records')

    return documents



def search(query,boost={}):

    documents = load_documents()

    text_fields   = ["exercise_name",
                    "type_of_activity",
                    "type_of_equipment",
                    "body_part",
                    "type",
                    "muscle_groups_activated",
                    "instructions"
                  ]

    index = minsearch.Index(
            text_fields=text_fields,
            keyword_fields=['id']
           )


    index.fit(documents)

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )


    return results




def build_prompt(query, search_results):

    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"


    prompt = prompt_template.format(question=query,context=context).strip()
    return prompt


def llm(prompt,model_choice):
    start_time = time.time()

    model = model_choice.split('/')[-1]

    response = open_client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content": prompt}]
    )
    
    answer = response.choices[0].message.content

    tokens = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }

    end_time = time.time()


    response_time = end_time - start_time


    return  answer, response_time , tokens


def rag(query,model):
    
    search_results = search(query)

    prompt = build_prompt(query,search_results)

    result = llm(prompt,model)

    return result

def rag_evaluation(answer_llm,query):
    
    
    prompt = eval_template.format(question=query,answer_llm=answer_llm)

    result = llm(prompt,'openai/gpt-4o')

    return result


def calculate_openai_cost(model, tokens):
    openai_cost = 0

    if model == 'openai/gpt-3.5-turbo':
        openai_cost = (tokens['prompt_tokens'] * 0.0015 + tokens['completion_tokens'] * 0.002) / 1000
    elif model in ['openai/gpt-4o', 'openai/gpt-4o-mini']:
        openai_cost = (tokens['prompt_tokens'] * 0.03 + tokens['completion_tokens'] * 0.06) / 1000

    return openai_cost


def get_answer(query,model):

    rag_answer, rag_response_time , rag_tokens = rag(query,model)
    eval_answer, eval_response_time , eval_tokens = rag_evaluation(rag_answer,query)

    response_time = rag_response_time + eval_response_time

    json_eval = json.loads(eval_answer)
    relevance =  json_eval['Relevance']
    explanation = json_eval['Explanation']


    openai_cost = calculate_openai_cost(model, rag_tokens)
    
    
    return {
        'answer': rag_answer,
        'response_time': response_time,
        'relevance': relevance,
        'relevance_explanation': explanation,
        'model_used': model,
        'prompt_tokens': rag_tokens['prompt_tokens'],
        'completion_tokens': rag_tokens['completion_tokens'],
        'total_tokens': rag_tokens['total_tokens'],
        'eval_prompt_tokens': eval_tokens['prompt_tokens'],
        'eval_completion_tokens': eval_tokens['completion_tokens'],
        'eval_total_tokens': eval_tokens['total_tokens'],
        'openai_cost': openai_cost
    }

   

