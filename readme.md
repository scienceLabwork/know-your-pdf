# Know Your PDF
Know Your PDF is an advanced system that leverages the power of Retrieval-Augmented Generation (RAG) to provide intelligent chat capabilities for interacting with PDF documents. It integrates four cutting-edge models: ChatGPT-4o, Gemma-8B, Mistral-8B, and Llama3-8B, each bringing unique strengths to the table.

![RAG Architecture](./rag_arc.png)

Features
1. RAG-Based Interaction
RAG combines the strength of retrieval mechanisms and generative models to provide accurate and contextually relevant responses. This allows Know Your PDF to:

    * Retrieve Information: Extract specific data from your PDFs.
    * Answer Questions: Provide detailed answers based on the document's content.
    * Summarize Content: Generate summaries for sections or entire documents.
    * Generate Insights: Offer insights and explanations derived from the text.
2. Multi-Model Support
By supporting four different models, Know Your PDF ensures versatility and robustness in handling various types of queries:

    * ChatGPT-4o: Excels in generating conversational and contextually rich responses.
    *Gemma-8B: Specialized in nuanced understanding and detailed explanations.
    * Mistral-8B: Known for its efficiency and accuracy in information retrieval.
    * Llama3-8B: Offers superior performance in processing large and complex texts.

## Evaluation
The performance of the models was evaluated on the following metrics:<br>
    1. Similarity Score: The cosine similarity between the generated Original Response and LLM Response.<br>
    ```
        SCORE: 0.83
    ```<br>
    File : [Evaluation](./evaluation_metrics_llm_org_similarity.csv)

## Installation
1. Clone this repository
2. Create a folder **.streamlit** and create a file **secrets.toml** and add your API KEYS<br>
``` python
HUGGINGFACEHUB_API_TOKEN="<KEY>"
OPEN_AI_KEY="<KEY>"
```
3. Open app.py, Change the model you like <br>
``` python
....
....
# - Mistral0.2-7B
# - Gemma1.1-7B
# - ChatGPT-4o
#USE THIS EXACT NAMES
model_choice = "ChatGPT-4o" #CHANGE ME

st.set_page_config(
    layout="wide",
    page_title="Know Your PDF",
    page_icon=":book:",
    initial_sidebar_state="expanded",
)
....
....
```

## Run Code
Just type this following in your terminal
``` bash
cd know-your-pdf
streamlit run app.py
```

## Contact Us
* shahrudra876@gmail.com<br>
* [linkedin](https://www.linkedin.com/in/rudra-shah-b044781b4/)<br>
* [Instagram](https://www.instagram.com/rudra_shah_/)
