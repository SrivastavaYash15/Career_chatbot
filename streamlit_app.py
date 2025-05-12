
import streamlit as st
import os
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

st.title("CareerBot")
st.write("Your Career Companion")

@st.cache_resource
def load_vector_db():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local('career_index', embedding_model,allow_dangerous_deserialization=True)
    return vector_db

vector_db = load_vector_db()
#api key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#defining the prompt template
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages = True, input_key = "question", output_key="text")
memory = st.session_state.memory

#setting up the chat model
chat_model = ChatGoogleGenerativeAI(
model="gemini-1.5-flash",
temperature=0,
max_output_tokens=1024,
google_api_key=GEMINI_API_KEY
)


#setting up the prompt template
prompt_temp = PromptTemplate.from_template(""" You are a helpful chatbot that answers career related questions.Users will ask you questions about their career and you will answer them. Do NOT asnwer if you dont know the answer. Keep the answer specific to the pdfs that have been provided to you. If you are not sure about the answer, say 'I am not sure about the answer'. Use your best judgment based on the context. Use prior conversation to stay in the context.
{history}
context = {context}
user_question: {question}""")


#setting up the llm chain
chain = LLMChain(
    llm= chat_model,
    prompt = prompt_temp,
    verbose = True,
    memory = memory
)

# Display chat history 
for msg in memory.chat_memory.messages:
    if msg.type == "human":
        st.markdown(f"**You:** {msg.content}")
    elif msg.type == "ai":
        st.markdown(f"**CareerBot:** {msg.content}")


user_input = st.text_input("Enter your query here:")

if user_input:
   with st.spinner("Searching for the answer..."):
    docs = vector_db.similarity_search(user_input, k=3)
    final_context_string = "\n\n".join(doc.page_content for doc in docs)
    response = chain.invoke({
        "context": final_context_string,
        "question": user_input
    })    
    st.write("Answer: ", response['text'])


    # Optional: Reset chat
if st.button("Reset Chat"):
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        input_key="question",
        output_key="text"
    )
    st.rerun()