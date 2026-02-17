A* 알고리즘 시각화 (Python)
+
+50×50 미로에서 시작점과 도착점을 설정하고, **A* 알고리즘이 탐색하는 과정 전체**를 색상으로 시각화합니다.
+
+## 기능
+
+1. 랜덤 미로 생성 (`50x50`, 벽 밀도 조절 가능)
+2. 시작점: `(1, 1)`, 도착점: `(size-2, size-2)`
+3. A* 알고리즘으로 최단 경로 탐색
+4. 색상 기반 **진행 과정** 시각화
+   - 흰색: 빈 칸
+   - 진회색: 벽
+   - 파란색: 열린 집합(Open Set)
+   - 회색: 닫힌 집합(Closed Set)
+   - 보라색: 현재 확장 중인 노드(Current)
+   - 초록색: 최종 경로
+   - 주황색: 시작점
+   - 빨간색: 도착점
+
+## 실행 방법
+
+```bash
+python3 a_star_visualization.py
+```
+
+옵션 예시:
+
+```bash
+python3 a_star_visualization.py --size 50 --wall-prob 0.30 --seed 7 --interval 0.01
+```
+
+## 결과 저장
+
+### 1) 탐색 과정을 애니메이션으로 저장 (권장)
+
+```bash
+python3 a_star_visualization.py --no-show --output process.gif
+```
+
+- `.gif`, `.mp4` 확장자는 **A* 진행 과정 전체**를 저장합니다.
+
+### 2) 최종 프레임(정적 이미지) 저장
+
+```bash
+python3 a_star_visualization.py --no-show --output final.png
+```
+
+- `.gif`, `.mp4` 이외 확장자는 마지막 프레임 이미지를 저장합니다.
+
+## 의존성
+
+- Python 3.10+
+- `matplotlib`
+- `numpy`
+
+설치:
+
+```bash
+pip install matplotlib numpy
+```
