# Vision

YOLO26 모델 변환 작업에 YOLOE 탐지·학습과 간단한 VAD를 추가한 브랜치입니다.

**[실행 방법과 인수인계 → yolo/README.md](yolo/README.md)**

1. 기존 YOLO26 → Rockchip RKNN 변환, YOLOE 프롬프트 고정 변환
2. YOLOE 텍스트·이미지 예시로 오픈 보캐뷸러리 탐지
3. 일반 YOLO 학습, YOLOE 미세 조정·텍스트/비전 프롬프트 학습
4. 탐지 결과와 시간 누적 규칙으로 영상 이상 후보 표시

기존 코드·모델 변환 결과·자습 파일은 보존했습니다. 이번 변경은 `yolo/`의 실행 코드와 문서에 집중합니다.
