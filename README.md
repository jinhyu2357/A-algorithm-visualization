codex가 README.md를 정리했음을 알립니다.  
a star 알고리즘에 벡터를 적용함.  
뭔지 모를 오류가 기다리고 있음  
실험버전  
맵제작(배경화면 있음) : https://jinhyu2357.github.io/A-algorithm-visualization/  

original A* 알고리즘 시각화 (Python + FastAPI)

## 기능

1. 랜덤 미로 생성 (가로 세로 조절 가능, 벽 밀도 조절 가능)
2. 시작점/도착점 랜덤 배치 (맨해튼 거리 최소 10칸 이상)
3. 벡터기반의 A* 알고리즘으로 최단 경로 탐색
4. 색상 기반 **진행 과정** 시각화
   - 흰색: 빈 칸
   - 진회색: 벽
   - 파란색: 열린 집합(Open Set)
   - 회색: 닫힌 집합(Closed Set)
   - 보라색: 현재 확장 중인 노드(Current)
   - 초록색: 최종 경로
   - 주황색: 시작점
   - 빨간색: 도착점
5. FastAPI 웹페이지에서 맵 제작 및 미리보기 기능

## 실행 방법

###  FastAPI 웹서버 실행

```bash
cd "A-algorithm-visualization"
.\.venv\Scripts\python -m uvicorn app:app --reload
```

브라우저에서 `http://127.0.0.1:8000` 접속 후,
그리드 크기/벽 밀도/시드를 입력해 미로 결과 이미지를 확인할 수 있습니다.


## 의존성

- Python 3.10+
- `matplotlib`
- `numpy`
- `fastapi`
- `uvicorn`
- `jinja2`
- `python-multipart`

설치:

```bash
pip install -r requirements.txt
```

## GitHub Pages 배포 가능 범위

- 현재 메인 기능(미로 생성 + A* 애니메이션 렌더링)은 `FastAPI` 서버와 Python 연산이 필요하므로 **GitHub Pages(정적 호스팅)에서 그대로 배포할 수 없습니다**.
- 대신 마우스 맵 에디터는 정적 페이지로 분리해 `docs/index.html`에 추가했습니다. 이 페이지는 GitHub Pages에서 바로 배포 가능합니다.

### 마우스 맵 에디터만 배포하는 방법

1. GitHub 저장소의 **Settings → Pages**로 이동
2. **Build and deployment**에서 Source를 **Deploy from a branch**로 선택
3. Branch를 현재 브랜치(또는 `main`) + `/docs` 폴더로 지정
4. 저장 후 배포 URL 접속

배포된 에디터에서 JSON을 내려받아, 로컬 FastAPI 앱의 `Custom Map (JSON)` 입력에 넣어 사용할 수 있습니다.
