import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import TavilySearchAPIRetriever
class retrievals :
    def __init__ (self):
    # ChromaDB가 저장된 폴더 경로
        self.PERSIST_DIRECTORY = "./chroma_db"

    # 임베딩 모델 정의 (임베딩에 사용했던 동일한 모델이어야 합니다)
        self.embeddings_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")


        

        # Re-rank 모델
        self.rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        self.cross_reranker = CrossEncoderReranker(model=self.rerank_model, top_n=2)


    def inspect_chroma_db(self,collection_name: str, query: str, k: int = 3):
        """
        지정된 컬렉션에 연결하고, 쿼리를 실행하여 결과를 출력합니다.
        """
        print(f"=== '{collection_name}' 컬렉션에 연결 중... ===")
        
        # ChromaDB 인스턴스 생성 (기존 DB 로드)
        # embedding_function이 있어야만 기존 데이터베이스를 올바르게 로드할 수 있습니다.
        db = Chroma(
            collection_name=collection_name, 
            persist_directory=self.PERSIST_DIRECTORY,
            embedding_function=self.embeddings_model
        )
        personal_db_retriever = ContextualCompressionRetriever(
            base_compressor=self.cross_reranker, 
            base_retriever=db.as_retriever(search_kwargs={"k":3}),
        )
        return personal_db_retriever.invoke(query)
    
    def web_retrievers(self, query :str):
        web_retriever = ContextualCompressionRetriever(
        base_compressor=self.cross_reranker, 
        base_retriever=TavilySearchAPIRetriever(k=10),
    )
        return web_retriever.invoke(query)
        
        
        

# # 사용 예시
# if __name__ == "__main__":
#     # 개인정보보호법 컬렉션 검색
#     inspect_chroma_db(
#         collection_name="personal_law", 
#         query="개인정보가 유출되면 어떻게 해야 하나요?"
#     )
    
#     # 근로기준법 컬렉션 검색
#     inspect_chroma_db(
#         collection_name="labor_law", 
#         query="미성년자 근로계약"
#     )

#     # 주택임대차보호법 컬렉션 검색
#     inspect_chroma_db(
#         collection_name="housing_law",
#         query="계약 갱신 요구권"
#     )