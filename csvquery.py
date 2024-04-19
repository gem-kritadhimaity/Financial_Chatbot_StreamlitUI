# from langchain_experimental.agents import create_csv_agent
# # from langchain.llms import OpenAI
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from huggingface_hub import InferenceClient 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
 
persist_directory = 'newcsvdb'
# persist_directory = 'csv_text_db'
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
 
 
def get_answer(context, question):
    client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1", token= "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")    # client = InferenceClient(model="meta-llama/Llama-2-7b-chat-hf", token=HF_TOKEN)
    res = "This is dummy values"
    res1 = ""
    try:
        prompt = f"""
            Use the following pieces of context to answer the user's question. If you don't know the answer reply 'Data Not Available'. 
                Context: {context} 
                User Question: {question} 
                Note: 
                1. net profit == Gross Profit.
                2. single line answer
                Answer :"""
        res = client.text_generation(prompt, max_new_tokens=500)

        prompt1 = f"""
        I have a question and an answer. You need to reply in a complete sentence. If numbers add $.
        Question : {question}
        Answer: {res}
        Reply:
        """

        res1 = client.text_generation(prompt1, max_new_tokens=200)
        print(prompt1)
 
    except:
        print('error')
       
    return res1
 
def hf_llm_qa(query):
    matching_docs = vectordb.similarity_search(query,k=6)
    context = []
    for docs in matching_docs:
        context.append("\n"+docs.page_content)
    # matching_docs

    answer = get_answer(', '.join(context),query)
    return answer

 
# print(hf_llm_qa("What is amazon's net profit for the latest available quarter in the database?"))
 
# #streamlit application
# st.title("Finance ChatBot")
# def process_query(query):
#     # Example logic to process the query
#     return hf_llm_qa(query.lower())
 
# def main():
 
#     # Accept user input
#     query = st.text_input('Enter your query here :')
 
#     if st.button('Submit'):
#         # Process the query
#         result = process_query(query)
       
#         # Display the result
#         st.write('Response:')
 
#         st.write(result.lower())
 
# if __name__ == '__main__':
#     main()