# from langchain_community.document_loaders import PyPDFLoader
import os

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_huggingface import HuggingFaceEmbeddings


# load pdf
# loader = PyPDFLoader("LENA_Manual_VM_v1.3.pdf")
# loader = UnstructuredHTMLLoader("C:/Users/won/IdeaProjects/tomcat/webapps/docs/connectors.xml")
# pages = loader.load_and_split()


# 특정 디렉토리 경로 설정
directory_path = "C:/dev/langchain/ai-study/tomcat_docs"

# 디렉토리 하위의 모든 HTML 파일을 로딩
pages = []
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith(".html"):  # HTML 파일 필터링
            file_path = os.path.join(root, file)
            loader = UnstructuredHTMLLoader(file_path)
            pages.extend(loader.load_and_split())  # 로딩 및 스플릿한 문서 추가

# print("# pages")
# print( pages)


# split pdf
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(pages)

# Embedding
# embeddings_model = OpenAIEmbeddings()
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Chroma DB 에 데이터 load
vectordb = Chroma.from_documents(texts, embeddings_model)

# OpenAI 에 db 와 함께 질의
question = "What is AJP connector?"

llm = CTransformers(
    model="C:/dev/langchain/ai-study/llama-2-7b.ggmlv3.q2_K.bin",
    model_type="llama"
)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
result = qa_chain.invoke({"query": question})

print("========== Answer ==========")
print(result["result"])
