# 실행 확인 (2026-10-08)

환경: Apple M1, Python 3.11, Ultralytics 8.4.174. 현장 정확도 평가와 실행 확인을 구분합니다.

| 항목 | 확인 결과 |
|---|---|
| 기존 코드 보존 | `origin/main`을 기반으로 변경. 기존 파일 삭제 없음 |
| 텍스트 탐지 | YOLOE-26n, OpenCV 공개 `vtest.avi` 5프레임 정상 처리, 첫 프레임 탐지 3개 |
| 비전 프롬프트 탐지 | 공개 `bus.jpg` 참조 박스 → 다른 영상 3프레임 정상 처리. 이 예시에서는 검출 0개, 탐지 품질 입증 아님 |
| 프롬프트 재사용 | NPZ 저장 후 변환·VAD에서 로드 |
| 변환 | YOLOE 프롬프트 고정 ONNX(opset 19, imgsz 64) 실제 생성. RKNN 변환 및 보드 실행은 미검증 |
| 텍스트 학습 | 공개 coco8-seg 일부, 1에폭, imgsz 64, best.pt 생성 |
| 고정 클래스 미세 조정 | 공개 coco8-seg 일부, 1에폭, imgsz 64, best.pt 생성 |
| 비전 프롬프트 학습 | 공개 coco8-seg 4장, 1에폭, imgsz 128, best.pt 생성. SAVPE 텐서 58개 중 54개가 변경된 것 확인 |
| VAD 연결 | 새 코드로 실제 영상 30프레임 정상 처리 |
| 시간 누적 규칙 | 지속·탐지 공백·중복 이벤트·잘못된 구역 설정 테스트 4개 통과 |

이전 VAD 실행에서는 100프레임(영상 10초)을 처리하고 9.9초에 crowding 후보를 기록했습니다. Mac 카메라는 OS 권한 때문에 열리지 않았습니다.

학습 확인은 소형 공개 데이터로 실행 경로를 검증한 것입니다. 생성된 best.pt를 현장용 모델이나 성능 개선 결과로 취급하지 않습니다. 현장 이미지/정답 라벨을 준비해 충분히 학습한 뒤 별도 검증하세요.

재현: 공개 coco8-seg 데이터 YAML 경로를 지정합니다. 아주 짧은 학습 확인은 gradient accumulation 때문에 업데이트가 생략되지 않도록 `--nbs`를 작게 지정하고 warmup을 끕니다.

```sh
python yolo/train.py --mode text --model yoloe-26n-seg.pt --data coco8-local.yaml \
  --epochs 1 --batch 2 --nbs 2 --warmup-epochs 0 --imgsz 128 --device cpu
python yolo/train.py --mode visual --model runs/vision/text/weights/best.pt --data coco8-local.yaml \
  --epochs 1 --batch 2 --nbs 2 --warmup-epochs 0 --imgsz 128 --device cpu
```

다음 확인: x86 Linux에서 RKNN 변환 → 보드 NPU 실행 → 현장 영상의 오탐/미탐 및 속도 → 학습 전후 비교.
