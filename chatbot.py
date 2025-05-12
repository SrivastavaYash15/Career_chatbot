import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf  #  import for PyMuPDF
import getpass
import os
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace
import os
import json
import re
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI



GEMINI_API_KEY = "AIzaSyCpmLCClHO9jbzJn7lMC4R8oORe2xtrfl8"

base_dir = "C:/Users/yashs/OneDrive/Desktop/Career_Chatbot/pdfs"  # Folder with PDFs

# Open the output file ONCE outside the loop to keep appending content from all PDFs
with open("output.txt", "w", encoding="utf-8") as out:
    for filename in os.listdir(base_dir):
        if filename.endswith(".pdf"):
            fullpath = os.path.join(base_dir, filename)  # Full path to the PDF
            doc = pymupdf.open(fullpath)  # Open the PDF
            for page in doc:
                text = page.get_text()  # This gives you a string — no need to encode
                out.write(text)  # Write the string directly
                out.write("\n\n--- End of Page ---\n\n")  # Optional: separator between pages
                


#creating chunks
loader = TextLoader("output.txt", encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_docs = splitter.split_documents(docs)
print(f"Number of chunks: {len(splitted_docs)}")
print(splitted_docs[0])
print(splitted_docs[3])




#  Initialize FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents([doc.page_content for doc in splitted_docs])
print(f"Number of embeddings: {len(embeddings)}")
print(f'Shape of first embedding: {embeddings[0].shape}')


#indexing to build the vector store
vecotor_db = FAISS.from_documents(documents = splitted_docs, embedding = embedding_model)


#using similarty search to find the most relevant chunks
"""query = ""
docs = vecotor_db.similarity_search(query, k=3)
print("Answer: ", docs)"""


#getting the llm
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_output_tokens=1024,
    google_api_key=GEMINI_API_KEY
)




from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

prompt_temp = ChatPromptTemplate.from_template(""" You are a helpful chatbot that answers career related questions.Users will ask you questions about their career and you will answer them. Do NOT asnwer if you dont know the answer. Keep the answer specific to the pdfs that have been provided to you. If you are not sure about the answer, say 'I am not sure about the answer'. Use your best judgment based on the context:
context = {context}
user_question: {question}""")

user_question = input("Please enter your question: ")

#choosing the chat model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_output_tokens=1024, 
    google_api_key=GEMINI_API_KEY
)

#creating a chain
chain = LLMChain(
    llm= chat_model,
    prompt = prompt_temp,
    verbose = True
)
final_context_string = "\n\n".join(doc.page_content for doc in docs)
response = chain.invoke({
    "context": final_context_string,
    "question": user_question
})
print(response['text'])

while True:
    user_question = input("Please enter your question: ")
    if user_question.lower() == "exit":
        break
    # Perform similarity search to find the most relevant chunks
    docs = vecotor_db.similarity_search(user_question, k=8) 
    final_context_string = "\n\n".join(doc.page_content for doc in docs)
    response = chain.invoke({
        "context": final_context_string,
        "question": user_question
    })
    print(response['text'])
     
    
   


