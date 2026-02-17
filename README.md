diff --git a/README.md b/README.md
index 022c840bf72dd8c3107182239e5c9eddf92f35ea..0627bbd69c48f68cb72fe316ebc16760dbceaf75 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,95 @@
-# A-algorithm-visualization
-Just, A* algorithm visualization
+# A* 알고리즘 시각화 (Python)
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
+
+## PR conflict(갈등)가 생기는 이유
+
+`PR conflict`는 **두 브랜치의 변경을 Git이 자동으로 합칠 수 없을 때** 발생합니다.
+즉, 코드가 반드시 잘못된 것이 아니라 "어느 버전을 남길지 사람이 결정해야 하는 상황"입니다.
+
+이 저장소 기준으로 자주 발생하는 케이스:
+
+1. **같은 파일의 같은 위치를 동시에 수정**
+   - 예: `README.md`, `a_star_visualization.py`의 동일한 문단/함수 라인을 각각 수정
+2. **한쪽은 파일 삭제/이동, 다른 쪽은 같은 파일 수정**
+3. **브랜치가 오래되어 `main` 최신 커밋을 반영하지 못한 상태에서 PR 생성**
+
+### 빠른 해결 방법 (rebase 권장)
+
+```bash
+git fetch origin
+git rebase origin/main
+# 충돌 발생 시 파일 열어서 <<<<<<<, =======, >>>>>>> 구간 정리
+git add <충돌 파일>
+git rebase --continue
+```
+
+### 대안 (merge)
+
+```bash
+git fetch origin
+git merge origin/main
+```
+
+### 충돌 예방 팁
+
+- PR 올리기 전에 `git fetch origin && git rebase origin/main`으로 최신화
+- 큰 변경은 작은 단위 커밋/PR로 나누기
+- 팀에서 자주 바꾸는 파일(`README.md`, 핵심 로직 파일)은 먼저 동기화 후 작업 시작
