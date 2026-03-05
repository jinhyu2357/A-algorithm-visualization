codex가 README.md를 정리했음을 알립니다.

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
