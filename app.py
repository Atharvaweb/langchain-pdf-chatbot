import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# 1. Streamlit page settings
# -----------------------------

st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 My AI Chatbot")


# -----------------------------
# 2. Create the AI model
# -----------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

general_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful and friendly AI assistant.
Use the previous conversation to understand follow-up questions.
Answer clearly and naturally."""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

general_chain = general_prompt | llm | StrOutputParser()



question_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Rewrite the latest user question as a standalone question
that can be understood without the previous conversation.

Use the chat history to understand follow-up questions.
Keep the original meaning.
Return only the rewritten question."""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

question_rewrite_chain = (
    question_rewrite_prompt
    | llm
    | StrOutputParser()
)


# -----------------------------
# 3. Create Prompt Template
# -----------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that answers questions about an uploaded PDF.

Use only the information in the PDF context.

When asked for work experience:
- List every distinct job or experience entry found in the context.
- For each entry, give the job title, company, and dates when available.
- Do not replace job titles with responsibilities.
- Do not omit entries just because they appear in different parts of the context.
- Do not invent missing details.
- Include responsibilities only when the user asks for them.

If the requested information is not present in the context, say:
"I couldn't find that information in the PDF."

PDF CONTEXT:
{context}
"""
    ),
    (
        "human",
        "{user_input}"
    )
])


# -----------------------------
# 

# -----------------------------
# 5. Chat memory
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# 6. PDF upload
# -----------------------------

st.sidebar.header("📄 Upload a PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type="pdf"
)

if uploaded_file:

    # Save uploaded PDF temporarily
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read PDF
    loader = PyPDFLoader("uploaded_file.pdf")
    documents = loader.load()

    # Split PDF into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
   )

    vectorstore = FAISS.from_documents(
    chunks,
    embeddings
   )

    retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
   )
    
    st.sidebar.success("✅ FAISS vector database created!")

    st.sidebar.success(
    f"PDF loaded! {len(documents)} pages found."
    )

    st.sidebar.info(
    f"Created {len(chunks)} text chunks."
    )
    
    # Create RAG Chains
    chain = (
        {
            
            "context": lambda x: "\n\n".join(
                doc.page_content
                for doc in retriever.invoke(
                    question_rewrite_chain.invoke({
                        "chat_history": x["chat_history"],
                        "user_input": x["user_input"]
                    })
                )
            ),

            "user_input": lambda x: question_rewrite_chain.invoke({
                "chat_history": x["chat_history"],
                "user_input": x["user_input"]
            })
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# -----------------------------
# 7. Clear Chat
# -----------------------------

if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# -----------------------------
# 8. Display previous messages
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# -----------------------------
# 9. Get user message
# -----------------------------

user_input = st.chat_input("Ask me anything...")


if user_input:

    with st.chat_message("user"):
        st.write(user_input)

    # Convert previous history
    chat_history = []

    for message in st.session_state.messages:

        if message["role"] == "user":
            chat_history.append(
                HumanMessage(content=message["content"])
            )

        else:
            chat_history.append(
                AIMessage(content=message["content"])
            )

    # Run chain
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            if uploaded_file:   
                response = chain.invoke({
                    "chat_history": chat_history,
                    "user_input": user_input
                })
            
            else:
                response = general_chain.invoke({
                    "chat_history": chat_history,
                    "user_input": user_input
                })


        st.write(response)

    # Save conversation
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })