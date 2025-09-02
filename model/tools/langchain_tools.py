# model/tools/langchain_tools.py

from langchain_core.tools import tool
from langchain_core.documents import Document
from typing import List

# vectorDB/retrieval.py 에서 설정된 Retriever들을 가져옵니다.
from vectorDB.retrieval import (
    personal_db_retriever, 
    labor_db_retriever, 
    housing_db_retriever, 
    web_retriever
)

@tool
def personal_law_search(query: str) -> List[Document]:
    """개인정보보호법 법률 조항을 검색합니다."""
    docs = personal_db_retriever.invoke(query)
    return docs if docs else [Document(page_content="관련 정보를 찾을 수 없습니다.")]

@tool
def labor_law_search(query: str) -> List[Document]:
    """근로기준법 법률 조항을 검색합니다."""
    docs = labor_db_retriever.invoke(query)
    return docs if docs else [Document(page_content="관련 정보를 찾을 수 없습니다.")]

@tool
def housing_law_search(query: str) -> List[Document]:
    """주택임대차보호법 법률 조항을 검색합니다."""
    docs = housing_db_retriever.invoke(query)
    return docs if docs else [Document(page_content="관련 정보를 찾을 수 없습니다.")]

@tool
def web_search(query: str) -> List[Document]:
    """데이터베이스에 없는 정보 또는 최신 정보를 웹에서 검색합니다."""
    docs = web_retriever.invoke(query)
    formatted_docs = []
    for doc in docs:
        formatted_docs.append(
            Document(
                page_content=f'<Document href="{doc.metadata["source"]}"/>\\n{doc.page_content}\\n</Document>',
                metadata={"source": "web search", "url": doc.metadata["source"]}
            )
        )
    return formatted_docs if formatted_docs else [Document(page_content="관련 정보를 찾을 수 없습니다.")]

# 에이전트에서 사용할 도구 목록
tools = [personal_law_search, labor_law_search, housing_law_search, web_search]