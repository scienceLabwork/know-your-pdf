import chainLLM
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

model_choice = "ChatGPT-4o"
data = chainLLM.pdf_data("policy-booklet-0923.pdf")
db = chainLLM.create_chunks(data)

def similarity_score(sentence1,sentence2):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embedding1 = embeddings_model.embed_query(sentence1)
    embedding2 = embeddings_model.embed_query(sentence2)
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
    similarity_score = round(similarity_score, 4)
    return similarity_score

df = pd.read_csv('evaluation_metrics.csv')
questions = df['Query'].tolist()
answers = df['Response'].tolist()
df_evaluation = {
    'Query': [],
    'Response': [],
    'Similarity_Score(-1 TO 1)': []
}
for i in range(len(questions)):
    query = questions[i]
    original_response = answers[i]
    llm_response = chainLLM.get_output(query, db, model_choice)
    similarity_score = similarity_score(original_response,llm_response)
    df_evaluation['Query'].append(query)
    df_evaluation['Response'].append(llm_response)
    df_evaluation['Similarity_Score(-1 TO 1)'].append(similarity_score)
    print(f"Query {i} done.")

df_evaluation = pd.DataFrame(df_evaluation)
df_evaluation.to_csv('evaluation_metrics.csv', index=False)