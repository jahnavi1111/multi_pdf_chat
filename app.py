import streamlit as st
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.set_page_config(page_title = "Chat with multiple PDFs", page_icon=":books:")

    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question about your uploaded documents:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"): #button becomes true only when users clicks on it
            with st.spinner("Processing"): #UI feature
                # get pdf text

                # get the text chunks

                # create vector store
                pass



if  __name__ == '__main__':
    main()