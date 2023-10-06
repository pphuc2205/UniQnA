from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title= "University Document Chatbot", page_icon=":books:")
    st.header("Ask Your PDF :books:")
    user_question = st.chat_input("Ask a question about  your PDF:")

    # tải tệp 
    with st.sidebar:
        st.subheader("Your documents")
        pdf = st.file_uploader("Upload PDF")

    # trích xuất văn bản 
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # chia thành nhiều đoạn nhỏ
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunk = text_splitter.split_text(text)

        # tạo phần nhúng
        # embeddings = OpenAIEmbeddings()
        # knowledge_base = FAISS.from_texts(chunk, embeddings)

        st.write(chunk)

        # hiển thị câu trả lời của người dùng
        # if user_question:
        #     docs = knowledge_base.similarity_search(user_question)

        #     llm = OpenAI()
        #     chain = load_qa_chain(llm, chain_type="stuff")
        #     with get_openai_callback() as cb:
        #         response = chain.run(input_documents=docs, question=user_question)
        #         print(cb)
           
        #     st.write(response)
if __name__ == '__main__':
    main()