from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings


# load pdf
loader = PyPDFLoader("LENA_Manual_VM_v1.3_mini.pdf")
pages = loader.load_and_split()

# split pdf
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=250,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(pages)

# Embedding
# embeddings_model = OpenAIEmbeddings()
embeddings_model = HuggingFaceEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")


# Chroma DB 에 데이터 load
vectordb = Chroma.from_documents(texts, embeddings_model)

# OpenAI 에 db 와 함께 질의
question = "LENA 의 구성요소 알려줘?"
# llm = ChatOpenAI(temperature=0)
from langchain_community.llms import CTransformers

llm = CTransformers(
    model="ggml-model-q2_k.bin",
    model_type="llama"
)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(search_kwargs={"k": 1}))
result = qa_chain.invoke({"query": question})

print(result)
