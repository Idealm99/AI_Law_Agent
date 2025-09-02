
from langchain_core.tools import tool
from typing import List
from vectorDB.retrieval import retrievals
from langchain_core.documents import Document

class tool_class:

    def __init__(self):
        self.tool_list = ['personal_law_search', 'labor_law_search', 'housing_law_search', 'web_search']
        self.searching_chroma=retrievals.inspect_chroma_db()
        self.searching_tavily=retrievals.web_retrievers()

    @tool
    def get_law_data(query: str):
        """
        특정 법률의 특정 조항에 대한 정보를 검색합니다.

        Args:
            query (str): 어떤 법률 조항을 검색할지 구분합니다. 
            (근로기준법 : personal_law , 개인정보보호법: labor_law, 주택임대차보호법: housing_law)
            
        """

    @tool
    def personal_law_search(self,query: str) -> List[Document]:
        """
        개인정보보호법 법률 조항을 검색합니다.
        """
        docs = self.searching_chroma('personal_law',query)

        if len(docs) > 0:
            return docs
        
        return [Document(page_content="관련 정보를 찾을 수 없습니다.")]

    @tool
    def labor_law_search(self,query: str) -> List[Document]:
        """
        근로기준법 법률 조항을 검색합니다.
        """
        docs = self.searching_chroma('labor_law',query)

        if len(docs) > 0:
            return docs
        
        return [Document(page_content="관련 정보를 찾을 수 없습니다.")]


    @tool
    def housing_law_search(self,query: str) -> List[Document]:
        """
        주택임대차보호법 법률 조항을 검색합니다.
        """
        docs = self.searching_chroma('housing_law',query)

        if len(docs) > 0:
            return docs
        
        return [Document(page_content="관련 정보를 찾을 수 없습니다.")]


    @tool
    def web_search(self,query: str) -> List[str]:
        """
        없다면 테빌리를 이용하여 법률 조항을 검색합니다.
        """

        docs = self.searching_tavily(query)

        formatted_docs = []
        for doc in docs:
            formatted_docs.append(
                Document(
                    page_content= f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>',
                    metadata={"source": "web search", "url": doc.metadata["source"]}
                )
            )

        if len(formatted_docs) > 0:
            return formatted_docs
        
        return [Document(page_content="관련 정보를 찾을 수 없습니다.")]