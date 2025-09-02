# vectorDB/retrieval.py

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import TavilySearchAPIRetriever

def setup_retrievers():
    """
    법률 문서 및 웹 검색을 위한 Retriever들을 설정하고 반환합니다.
    """
    # 문서 임베딩 모델
    embeddings_model = OllamaEmbeddings(model="bge-m3")

    # Re-rank 모델
    rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=rerank_model, top_n=2)

    # 각 법률별 Chroma DB Retriever 설정
    def create_db_retriever(collection_name):
        db = Chroma(
            embedding_function=embeddings_model,
            collection_name=collection_name,
            persist_directory="./chroma_db",
        )
        base_retriever = db.as_retriever(search_kwargs={"k": 5})
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )

    personal_retriever = create_db_retriever("personal_law")
    labor_retriever = create_db_retriever("labor_law")
    housing_retriever = create_db_retriever("housing_law")

    # 웹 검색 Retriever
    web_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=TavilySearchAPIRetriever(k=10),
    )
    
    return personal_retriever, labor_retriever, housing_retriever, web_retriever

# Retriever 인스턴스 생성
personal_db_retriever, labor_db_retriever, housing_db_retriever, web_retriever = setup_retrievers()