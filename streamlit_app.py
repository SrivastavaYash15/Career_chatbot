
import streamlit as st
import os
import requests
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

st.markdown("""
    <style>
    .stChatMessage {
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        margin-bottom: 0.5rem;
    }
    .stChatMessage.user {
        background-color: #DCF8C6;
        align-self: flex-end;
    }
    .stChatMessage.assistant {
        background-color: #F1F0F0;
    }
    </style>
""", unsafe_allow_html=True)

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
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
print(SERPAPI_KEY)
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
#setting up the chain for web search 
web_prompt = PromptTemplate.from_template("""You are a helpful AI assistant who answers career-related questions using search results from the web.

Only use the information provided below. Do not rely on prior knowledge or assumptions. Keep the response concise and relevant.

If the web information is insufficient to answer accurately, say: "I'm not confident enough to answer this question."

{history}
Web Results:
{web_context}

User Question:
{question}
""")

#setting up the llm chain
web_chain = LLMChain(
    llm=chat_model,
    prompt= web_prompt,
    verbose=True,
    memory=memory
)


#creating function to perform a web search
def get_web_results(query, api_key, k=3):
    url = "https://serpapi.com/search"  # The base URL for SerpAPI
    params = {
        'q': query,
        'engine': 'google',
        'api_key': api_key,
        'num': k, 
        }
    response = requests.get("https://serpapi.com/search.json", params=params)
    results = response.json().get('organic_results', [])
    snippets = [r.get('snippet', '') for r in results if r.get('snippet')]
    return "\n".join(snippets)
    # Perform the web search
# Display chat history
for msg in memory.chat_memory.messages:
    with st.chat_message("user" if msg.type == "human" else "assistant"):
        st.markdown(msg.content)
    


user_input = st.chat_input("Enter your query here:")
if user_input is not None and user_input.strip() != "":
    with st.spinner("Searching for the answer..."):
        docs_with_scores = vector_db.similarity_search_with_score(user_input, k=3)
        final_context_string = "\n\n".join(doc.page_content for doc, _ in docs_with_scores)

    response = chain.invoke({
        "context": final_context_string,
        "question": user_input
    })

    # Check if the model replied with "not sure"
    if "not sure" in response['text'].lower():
        web_context = get_web_results(user_input, SERPAPI_KEY)
        response = web_chain.invoke({
            "web_context": web_context,
            "question": user_input
        })
        with st.chat_message("assistant"):
            st.markdown(f" I searched the web and found:\n\n{response['text']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(response['text'])




    # Optional: Reset chat
if st.button("Reset Chat"):
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        input_key="question",
        output_key="text"
    )
    st.rerun()

