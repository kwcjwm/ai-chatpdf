__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from dotenv import load_dotenv
# load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

# 제목
st.title("Chat with PDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("Upload PDF file",type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath,"wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    
    # split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300, # 몇자 단위로 자르는가 
        chunk_overlap  = 20, # 오버랩 중복 허용
        length_function = len, #길이 함수 뭐쓸까
        is_separator_regex = False,
    )


    # loader
    # loader = PyPDFLoader("./levSpecSampling.pdf")
    # pages = loader.load_and_split()

    texts = text_splitter.split_documents(pages)

    # embeddings
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    # Question
    # question = "논문 핵심 내용을 한글로 설명해줘."
    # llm = ChatOpenAI(temperature=0)
    # retriever_from_llm = MultiQueryRetriever.from_llm(
    #     retriever=db.as_retriever(), llm=llm
    # )
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요.')

    # docs = retriever_from_llm.get_relevant_documents(query=question)
    # print(len(docs))
    # print(docs)

    # 질문하기
    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])

