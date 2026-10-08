# 문제 해결

[문서 홈](../README.md)

| 증상 | 확인 및 조치 |
|---|---|
| `Cannot open source` | 파일 경로, 카메라 번호, 장치 연결 및 OS 카메라 권한 확인 |
| `Video FPS unavailable` | 영상 FPS 메타데이터 확인. 유효한 FPS를 가진 파일로 준비 |
| RKNN 변환 환경 오류 | x86 Linux에서 실행. Mac은 `--format onnx`로 내보내기만 확인 |
| INT8 변환 거부 | segmentation 모델에는 `--quantize 16` 사용 |
| 동적 프롬프트 오류 | RKNN 대신 원본 YOLOE `.pt` 사용. 프롬프트 옵션은 하나만 지정 |
| 프로필 아키텍처 불일치 | 해당 모델에서 프로필을 다시 생성 |
| 참조 이미지 오류 | JSON 기준 상대 경로와 박스 좌표, 클래스 번호 확인 |
| 비전 학습의 fused 체크포인트 오류 | 고정 클래스 미세 조정 결과 대신 원본 또는 `text` 결과 사용 |
| GPU 메모리 부족 | `--batch` 또는 `--imgsz` 감소 |
| 매우 짧은 학습에서 가중치 변화 없음 | warmup·gradient accumulation 확인. 실행 검증용으로 `--warmup-epochs 0 --nbs 2 --batch 2` 사용 |
| 화면 표시 실패 | GUI 없는 서버에서는 `--no-display` 사용 |
| 이벤트 파일 열기 실패 | 출력 폴더 존재 여부와 쓰기 권한 확인 |
| 사람이 있어도 VAD 반응 없음 | 클래스명이 `person_labels`와 일치하는지, 탐지 점수와 지속시간 확인 |
| 보드 RKNN 로딩·추론 실패 | 대상 칩, 변환 도구 버전, NPU 드라이버와 런타임 호환성 확인 |

## 명령 도움말

```sh
python yolo/predict.py --help
python yolo/train.py --help
python yolo/yolo26n-rknn/install.py --help
python yolo/vad/detect.py --help
```

## 규칙 검증

```sh
python -m unittest discover -s yolo/tests -v
```

공개 데이터 실행 결과와 미검증 항목은 [검증 기록](../yolo/VALIDATION.md)에 있습니다.
