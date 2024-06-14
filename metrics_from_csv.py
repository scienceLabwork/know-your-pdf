import chainLLM
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def similarity_score(sentence1,sentence2):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embedding1 = embeddings_model.embed_query(sentence1)
    embedding2 = embeddings_model.embed_query(sentence2)
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
    similarity_score = round(similarity_score, 4)
    return similarity_score

df = pd.read_csv('llm_Response.csv')
new_df = {
    'Original Response': [],
    'LLM Response': [],
    'Similarity Score': []
}
for i in range(len(df)):
    original_response = df['Original Response'][i]
    llm_response = df['LLM Response'][i]
    similarity = similarity_score(original_response,llm_response)
    new_df['Original Response'].append(original_response)
    new_df['LLM Response'].append(llm_response)
    new_df['Similarity Score'].append(similarity)
    print(f"Query {i} done.")
new_df = pd.DataFrame(new_df)
new_df.to_csv('evaluation_metrics_llm_org_similarity.csv', index=False)