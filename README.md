# A* 알고리즘 시각화 (Python)

50×50 미로에서 시작점과 도착점을 설정하고, A* 알고리즘 탐색 과정을 색상으로 시각화하는 예제입니다.

## 기능

1. 랜덤 미로 생성 (`50x50`, 벽 밀도 조절 가능)
2. 시작점: `(1, 1)`, 도착점: `(size-2, size-2)`
3. A* 알고리즘으로 최단 경로 탐색
4. 색상 기반 진행 상태 시각화
   - 흰색: 빈 칸
   - 진회색: 벽
   - 파란색: 열린 집합(Open Set)
   - 회색: 닫힌 집합(Closed Set)
   - 초록색: 최종 경로
   - 주황색: 시작점
   - 빨간색: 도착점

## 실행 방법

```bash
python3 a_star_visualization.py
```

옵션 예시:

```bash
python3 a_star_visualization.py --size 50 --wall-prob 0.30 --seed 7 --interval 0.01
```

GUI 없이 실행하고 결과 이미지를 저장하려면:

```bash
python3 a_star_visualization.py --no-show --output artifact.png
```

## 의존성

- Python 3.10+
- `matplotlib`
- `numpy`

설치:

```bash
pip install matplotlib numpy
```
