import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# ChromaDB가 저장된 폴더 경로
PERSIST_DIRECTORY = "./chroma_db"

# 임베딩 모델 정의 (임베딩에 사용했던 동일한 모델이어야 합니다)
embeddings_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")

def inspect_chroma_db(collection_name: str, query: str, k: int = 2):
    """
    지정된 컬렉션에 연결하고, 쿼리를 실행하여 결과를 출력합니다.
    """
    print(f"=== '{collection_name}' 컬렉션에 연결 중... ===")
    
    # ChromaDB 인스턴스 생성 (기존 DB 로드)
    # embedding_function이 있어야만 기존 데이터베이스를 올바르게 로드할 수 있습니다.
    db = Chroma(
        collection_name=collection_name, 
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings_model
    )
    
    # 유사도 검색 수행
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    
    print(f"=== '{query}' 쿼리에 대한 검색 결과 (상위 {k}개) ===")
    if docs:
        for i, doc in enumerate(docs):
            print(f"\n--- 문서 {i+1} ---")
            print(f"출처: {doc.metadata.get('name')} - {doc.metadata.get('chapter', '조문')}")
            print(f"내용:\n{doc.page_content}")
            print("-" * 20)
    else:
        print("검색된 문서가 없습니다.")

# 사용 예시
if __name__ == "__main__":
    # 개인정보보호법 컬렉션 검색
    inspect_chroma_db(
        collection_name="personal_law", 
        query="개인정보가 유출되면 어떻게 해야 하나요?"
    )
    
    # 근로기준법 컬렉션 검색
    inspect_chroma_db(
        collection_name="labor_law", 
        query="미성년자 근로계약"
    )

    # 주택임대차보호법 컬렉션 검색
    inspect_chroma_db(
        collection_name="housing_law",
        query="계약 갱신 요구권"
    )