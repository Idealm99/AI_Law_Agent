# 필요한 라이브러리 임포트
import os
import re
from glob import glob
from textwrap import dedent
import uuid
import warnings

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# .env 파일 로드
load_dotenv()

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# 임베딩 모델 정의 (Hugging Face 모델 사용)
# 'BAAI/bge-m3'는 다국어(한국어 포함) 성능이 우수한 모델입니다.
# HuggingFaceBgeEmbeddings 클래스를 사용하려면 'sentence-transformers' 패키지가 필요합니다.
embeddings_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")

# 법률 텍스트 파싱 함수
def parse_law(law_text):
    """
    법률 텍스트를 서문, 장, 조문, 부칙으로 분리하여 딕셔너리로 반환합니다.
    """
    preamble_pattern = r'^(.*?)(?=제1장|제1조)'
    preamble = re.search(preamble_pattern, law_text, re.DOTALL)
    if preamble:
        preamble = preamble.group(1).strip()
    else:
        preamble = ""

    chapter_pattern = r'(제\d+장\s+.+?)\n((?:제\d+조(?:의\d+)?(?:\(\w+\))?.*?)(?=제\d+장|부칙|$))'
    chapters = re.findall(chapter_pattern, law_text, re.DOTALL)

    appendix_pattern = r'(부칙.*)'
    appendix = re.search(appendix_pattern, law_text, re.DOTALL)
    if appendix:
        appendix = appendix.group(1).strip()
    else:
        appendix = ""

    parsed_law = {'서문': preamble, '장': {}, '부칙': appendix}

    article_pattern = r'(제\d+조(?:의\d+)?\s*\([^)]+\).*?)(?=제\d+조(?:의\d+)?\s*\([^)]+\)|$)'

    if chapters:
        for chapter_title, chapter_content in chapters:
            articles = re.findall(article_pattern, chapter_content, re.DOTALL)
            parsed_law['장'][chapter_title.strip()] = [article.strip() for article in articles]
    else:
        main_text = re.sub(preamble_pattern, '', law_text, flags=re.DOTALL)
        main_text = re.sub(appendix_pattern, '', main_text, flags=re.DOTALL)
        articles = re.findall(article_pattern, main_text, re.DOTALL)
        parsed_law['조문'] = [article.strip() for article in articles]

    return parsed_law

def process_pdf_and_embed(pdf_file_path):
    """
    PDF 파일을 로드, 파싱, 정리하여 ChromaDB에 임베딩합니다.
    """
    # 파일 이름에서 법률 이름 추출 및 컬렉션 이름 설정
    law_name_raw = os.path.basename(pdf_file_path).split('(')[0].strip()
    law_name_base = law_name_raw.replace('법률', '').replace(' ', '').replace('개인정보', '개인정보 보호법')
    law_name = law_name_base if law_name_base else law_name_raw
    collection_name_map = {
        '근로기준법': 'labor_law',
        '개인정보보호법': 'personal_law',
        '주택임대차보호법': 'housing_law'
    }
    collection_name = collection_name_map.get(law_name.replace(' ', ''), 'default_law')

    print(f"=== {law_name} ({collection_name}) 임베딩 시작 ===")

    # PDF 로드
    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load()

    # 페이지 텍스트 병합 및 불필요한 헤더 삭제
    text_for_delete_pattern = r"법제처\s+\d+\s+국가법령정보센터\n" + law_name
    law_text = "\n".join([re.sub(text_for_delete_pattern, "", p.page_content).strip() for p in pages])
    
    # 법률 파싱
    parsed_law = parse_law(law_text)
    
    final_docs = []

    # 파싱된 내용으로 Document 객체 생성
    if '장' in parsed_law and parsed_law['장']:
        for chapter, articles in parsed_law['장'].items():
            metadata = {
                "source": pdf_file_path,
                "chapter": chapter,
                "name": law_name
            }
            for article in articles:
                content = dedent(f"""
                [법률정보]
                다음 조항은 {metadata['name']} {metadata['chapter']}에서 발췌한 내용입니다.

                [법률조항]
                {article}
                """)
                final_docs.append(Document(page_content=content, metadata=metadata))
    elif '조문' in parsed_law and parsed_law['조문']:
        metadata = {
            "source": pdf_file_path,
            "name": law_name
        }
        for article in parsed_law['조문']:
            content = dedent(f"""
            [법률정보]
            다음 조항은 {metadata['name']}에서 발췌한 내용입니다.

            [법률조항]
            {article}
            """)
            final_docs.append(Document(page_content=content, metadata=metadata))
    else:
        print("경고: 파싱된 내용이 없습니다.")
        return

    # Chroma 인덱스 생성 (기존 데이터 삭제 후 새로 생성)
    _ = Chroma.from_documents(
        documents=final_docs, 
        embedding=embeddings_model,   
        collection_name=collection_name,
        persist_directory="./chroma_db",
    )
    
    print(f"총 {len(final_docs)}개의 문서를 {collection_name} 컬렉션에 임베딩했습니다.")
    print("--------------------------------------------------")


if __name__ == "__main__":
    # pdf 파일 목록을 확인
    pdf_files = glob(os.path.join('data', '*.pdf'))
    
    if not pdf_files:
        print("data 폴더에 PDF 파일이 없습니다. 파일을 추가해 주세요.")
    else:
        for file_path in pdf_files:
            process_pdf_and_embed(file_path)