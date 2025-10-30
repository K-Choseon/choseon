# 조선·해양 매뉴얼 기반 RAG 챗봇 (Streamlit)

## 실행 방법
1) 의존성 설치
```
pip install -r requirements.txt
```

2) 환경변수 설정
- 프로젝트 루트에 `.env` 파일 생성
```
OPENAI_API_KEY=sk-...
```

3) 앱 실행
```
streamlit run app.py
```

## 폴더 구조(생성 예정)
- `app.py`: 메인(채팅) 페이지
- `pages/01_퀴즈.py`: 퀴즈 페이지
- `rag/`: 파서/임베딩/인덱스/저장소/RAG/퀴즈 모듈
- `data/`: 로컬 저장소 (업로드 매뉴얼/인덱스/메타)

## 주의
- 업로드 가능한 파일 형식: PDF
- 로컬 저장만 사용, 로그인 기능 없음
