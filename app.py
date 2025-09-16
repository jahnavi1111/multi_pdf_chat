import streamlit as st #for GUI
from dotenv import load_dotenv #load keys
from PyPDF2 import PdfReader #for handling pdfs
from langchain.text_splitter import CharacterTextSplitter #for text splitting
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) # can access pages of pdf
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function=len
        )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings() #embedding model object
    embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl') #https://huggingface.co/hkunlp/instructor-xl
    vector_store = FAISS.from_texts(texts = text_chunks, embedding = embeddings) 
    #calls the embedding model on each tex chunk to generate embeddings and stores them in FAISS vector store
    return vector_store

def get_conversation_chain(vector_store):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", 
                         model_kwargs={"temperature":0.5, "max_length":512},
                         task="text2text-generation"  # Required argument
                         )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

        

def main():
    load_dotenv()
    st.set_page_config(page_title = "Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        # for when application re-runs itself (same session while the appn is open), set it to None if it's not being initialized
        st.session_state.conversation = None 
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your uploaded documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"): #button becomes true only when users clicks on it
            with st.spinner("Processing"): #UI feature
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # create vector store
                vector_store = get_vector_store(text_chunks)

                #create conversation chain
                #conversation = get_conversation_chain(vector_store)
                st.session_state.conversation = get_conversation_chain(vector_store) # use to make conversation persistent over time, and also would be available outside of it's scope


if  __name__ == '__main__':
    main()