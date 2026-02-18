A* 알고리즘 시각화 (Python + FastAPI)

50×50 미로에서 시작점과 도착점을 설정하고, **A* 알고리즘이 탐색하는 과정 전체**를 색상으로 시각화합니다.

## 기능

1. 랜덤 미로 생성 (`50x50`, 벽 밀도 조절 가능)
2. 시작점/도착점 랜덤 배치 (맨해튼 거리 최소 10칸 이상)
3. A* 알고리즘으로 최단 경로 탐색
4. 색상 기반 **진행 과정** 시각화
   - 흰색: 빈 칸
   - 진회색: 벽
   - 파란색: 열린 집합(Open Set)
   - 회색: 닫힌 집합(Closed Set)
   - 보라색: 현재 확장 중인 노드(Current)
   - 초록색: 최종 경로
   - 주황색: 시작점
   - 빨간색: 도착점
5. FastAPI 웹페이지에서 파라미터를 입력하고 결과 이미지를 즉시 미리보기

## 실행 방법

### 1) CLI 시각화 실행

```bash
python3 a_star_visualization.py
```

옵션 예시:

```bash
python3 a_star_visualization.py --size 50 --wall-prob 0.30 --seed 7 --interval 0.01
```

### 2) FastAPI 웹서버 실행

```bash
uvicorn app:app --reload
```

브라우저에서 `http://127.0.0.1:8000` 접속 후,
그리드 크기/벽 밀도/시드를 입력해 미로 결과 이미지를 확인할 수 있습니다.

## 결과 저장

### 1) 탐색 과정을 애니메이션으로 저장 (권장)

```bash
python3 a_star_visualization.py --no-show --output process.gif
```

- `.gif`, `.mp4` 확장자는 **A* 진행 과정 전체**를 저장합니다.

### 2) 최종 프레임(정적 이미지) 저장

```bash
python3 a_star_visualization.py --no-show --output final.png
```

- `.gif`, `.mp4` 이외 확장자는 마지막 프레임 이미지를 저장합니다.

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
