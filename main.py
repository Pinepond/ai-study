from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI




# load pdf
loader = PyPDFLoader("LENA_Manual_VM_v1.3.pdf")
pages = loader.load_and_split()

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
# from langchain_community.embeddings.sentence_transformer import (
#     SentenceTransformerEmbeddings,
# )
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings_model = OpenAIEmbeddings()

# Chroma DB 에 데이터 load
vectordb = Chroma.from_documents(texts, embeddings_model)

# OpenAI 에 db 와 함께 질의
question = "LENA 에 대해서 소개해줘?"
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
result = qa_chain({"query": question})

print(result)
