# CareerBot: Your Career Companion

CareerBot is an intelligent chatbot designed to assist users with career-related queries using PDF knowledge and real-time web search. Built with Streamlit, LangChain, FAISS, HuggingFace Embeddings, and Google's Gemini 1.5 Flash model, CareerBot gives personalized answers backed by trusted documents and the internet.

---

## Features

-  **PDF-based QA**: Answers questions by extracting context from indexed PDF files.
-  **Web Fallback**: If the answer isn't found in the PDFs, it uses real-time Google search via SerpAPI.
- **Chat Interface**: Clean and conversational UI using `st.chat_input` and `st.chat_message` (ChatGPT-like).
- **Conversation Memory**: Remembers previous queries to provide context-aware responses.
-  **Fast Vector Search**: Powered by FAISS and `sentence-transformers/all-MiniLM-L6-v2`.

---

##  Project Structure

CareerBot/
├── .env # Environment variables (API keys)
├── career_index/ # FAISS vector index files
├── pdfs/ # Source PDFs for chatbot knowledge
├── chatbot.py # (Optional helper module)
├── output.txt # Output logs (if any)
├── streamlit_app.py #  Main Streamlit app
└── README.md #  You're reading it!



---

##  Setup Instructions

### 1. Clone the repo

git clone https://github.com/YourUsername/CareerBot.git
cd CareerBot

### 2. Create & activate environment 
conda create -n careerbot-env python=3.10
conda activate careerbot-env

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add your .env file
GEMINI_API_KEY=your_google_gemini_key
SERPAPI_KEY=your_serpapi_key

### 5. Run the App
streamlit run streamlit_app.py

----

##  How It Works
PDFs are embedded into a vector database using sentence-transformers + FAISS.

When a user asks a question:

CareerBot checks the most relevant PDF chunks.

If a confident answer is found → returns it using Gemini LLM.

Otherwise → performs a web search via SerpAPI and answers from that.

----

##  Tech Stack
LLM: Google Gemini 1.5 Flash via langchain-google-genai

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector DB: FAISS

Frontend: Streamlit

Search: SerpAPI (Google Search JSON API)

Memory: LangChain's ConversationBufferMemory

----

##  To-Do / Future Improvements
Add source highlighting or document links

Enable PDF upload during runtime

Improve web search filtering

Save chat history persistently

Add UI themes or personalization

----

##  Acknowledgments
LangChain

Hugging Face Sentence Transformers

Google Gemini

SerpAPI

Streamlit
