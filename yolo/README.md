# Vision 실행·인수인계

기존 `yolo/yolo26n-rknn/install.py` 변환 코드에 탐지·학습·VAD를 추가했습니다. 기존 파일과 자습 폴더는 보존했고, 아래 작업은 `yolo/` 안에서 이어갑니다. 명령은 저장소 루트에서 실행합니다.

## 준비

```sh
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r yolo/requirements.txt
source .venv/bin/activate
```

최초 실행은 모델과 텍스트 인코더를 다운로드합니다. 카메라는 실행 앱의 OS 권한이 필요합니다. `q` 또는 Ctrl+C로 종료합니다. `--source 0` 대신 영상 경로나 스트림 주소를 넣을 수 있습니다.

## 1. Rockchip에서 돌리기

**변환은 x86 Linux, 실행은 Rockchip 보드**에서 합니다. 기존 코드의 YOLO26 → RKNN INT8 변환은 그대로 지원합니다.

```sh
# x86 Linux
uv pip install --python .venv/bin/python -r yolo/requirements-export.txt
python yolo/yolo26n-rknn/install.py --model yolo26n.pt --target rk3588 --quantize 8 --data coco8.yaml

# YOLOE: 탐지할 텍스트를 고정한 FP16 변환
python yolo/yolo26n-rknn/install.py --model yoloe-26n-seg.pt --quantize 16 --prompts person bottle
```

변환 결과는 모델 옆의 `*_rknn_model/` 폴더입니다. `.rknn`과 `metadata.yaml`을 함께 보드에 복사합니다. 보드에는 호환되는 RKNN 런타임과 Ultralytics가 필요합니다.

```sh
# Rockchip 보드 (Python 3.11 환경과 보드 NPU 드라이버 필요)
uv pip install --python .venv/bin/python -r yolo/requirements-rockchip.txt
python yolo/predict.py --model ./yolo26n_rknn_model --source 0
```

Mac에서는 `--format onnx`로 중간 ONNX 내보내기를 확인할 수 있습니다. RKNN 변환 완료를 뜻하지 않습니다. INT8은 detection 모델만 지원하므로 YOLOE segmentation에는 FP16을 사용합니다. 실제 배포에는 현장 이미지로 calibration하세요.

**RKNN으로 변환한 뒤에는 프롬프트를 바꿀 수 없습니다.** 변경하려면 원본 `.pt`에서 프롬프트를 설정하고 다시 변환합니다. 동적으로 바꾸려면 아래 YOLOE `.pt` 경로를 사용하세요.

## 2. YOLOE 오픈 보캐뷸러리 탐지

```sh
# 텍스트로 대상 지정
python yolo/predict.py --model yoloe-26n-seg.pt --source 0 --prompts person bottle "dog robot"

# 이미지 예시로 대상 지정
python yolo/predict.py --model yoloe-26n-seg.pt --source sample.mp4 --visual yolo/visual.json
```

`yolo/visual.example.json`을 `visual.json`으로 복사해 수정합니다. `image`는 참조 이미지 경로, `bboxes`는 그 이미지의 픽셀 좌표 `[x1,y1,x2,y2]`, `cls`는 0부터 이어지는 클래스 번호, `names`는 표시할 이름입니다. 이미지 경로는 JSON 파일 기준입니다.

프롬프트 프로필은 `--save-embeddings /tmp/prompts.npz`로 저장하고, 다음에는 `--embeddings /tmp/prompts.npz`로 재사용할 수 있습니다. 변환 코드도 `--visual`과 `--embeddings`를 지원합니다. 참조 이미지에는 학습이 필요하지 않습니다.

## 3. 학습시키기

현장 이미지와 라벨을 준비하고 `yolo/dataset.example.yaml`을 복사해 경로와 클래스명을 수정합니다. 영상은 프레임 이미지로 준비합니다. 데이터와 학습 가중치는 Git에 올리지 않습니다.

```sh
# 일반 YOLO26: bbox 라벨로 학습
python yolo/train.py --mode yolo --model yolo26n.pt --data dataset.yaml --device 0

# YOLOE: 데이터셋 클래스명으로 고정 클래스 미세 조정 (polygon 라벨)
python yolo/train.py --mode finetune --model yoloe-26n-seg.pt --data dataset-seg.yaml --device 0

# YOLOE 텍스트-이미지 학습: 클래스 이름과 이미지/polygon 라벨
python yolo/train.py --mode text --model yoloe-26n-seg.pt --data dataset-seg.yaml --device 0

# 비전 프롬프트 인코더 학습: text 단계의 unfused best.pt 사용
python yolo/train.py --mode visual --model runs/vision/text/weights/best.pt --data dataset-seg.yaml --device 0
```

`--device 0`은 CUDA GPU입니다. Mac은 `--device mps`, CPU는 `--device cpu`를 사용합니다. `--epochs`, `--batch`, `--imgsz`로 작업량을 조절합니다. 결과 경로는 실행 끝에 출력됩니다. 같은 모드를 여러 번 실행하면 이름에 번호가 붙을 수 있습니다.

`finetune`은 클래스명을 모델에 고정합니다. 이후 자유로운 텍스트/비전 프롬프트 학습에는 `text` 결과를 사용합니다. `visual`은 나머지 레이어를 고정하고 SAVPE만 학습합니다. 텍스트만 입력한다고 학습되는 것은 아니며, 이미지와 bbox/polygon 정답이 필요합니다. 세그멘테이션 모드는 polygon 라벨을 준비하세요.

문장-영역 grounding 데이터가 있다면 `text`/`visual`의 `--data`에 공식 형식의 YAML(`train: {yolo_data: [...], grounding_data: [{img_path: ..., json_file: ...}]}`, `val: {yolo_data: [...]}`)을 줄 수도 있습니다. 이 경로는 별도 데이터 준비가 필요합니다.

## 4. 간단한 VAD

```sh
python yolo/vad/detect.py --model yoloe-26n-seg.pt --source 0 \
  --prompts person "standing person" "person lying on the ground"
# 보드에서는 --prompts 없이 정적 RKNN 모델 폴더를 지정
```

가로로 긴 사람 박스/누운 사람은 낙상 의심, 사람 박스 5개 초과는 군중 후보입니다. 조건을 2초 누적하고 0.3초 탐지 공백을 허용합니다. `yolo/vad/config.json`에서 조절합니다. 제한 구역은 기본 꺼짐이며 비율 좌표 `[0.2,0.2,0.8,0.8]` 등으로 지정합니다. 비전 프롬프트도 지원하며 `names`가 VAD의 `person_labels`와 맞아야 합니다.

`--events /tmp/vad-events.jsonl`로 이벤트를 기록합니다. 이는 규칙 기반 후보이며 실제 낙상/위험을 확정하지 않습니다. 사람별 추적은 없고, 박스 중복·오탐·미탐 영향을 받습니다. 처리가 느려도 매번 조건이 검출되면 누적합니다. 관측 사이에 일어난 일은 알 수 없습니다. Bayesian은 관련 설계 메모 수준이며 확률 계산은 넣지 않았습니다.

## 검증과 남은 일

실제 샘플 영상 100프레임 처리와 `crowding` 이벤트 출력은 앞선 VAD 실행에서 확인했습니다. 새 브랜치 검증 결과는 `VALIDATION.md`에 적습니다. 현장 학습 정확도, RKNN 변환 결과와 보드 NPU 실행은 별도로 확인해야 합니다.

기존 변환 결과물은 `yolo/yolo26n-rknn/yolo26n_rknn_model/`에 보존했습니다. 과거 파일 존재만으로 현재 보드 실행이 확인된 것은 아닙니다.

API 기준: [YOLOE 공식 문서](https://docs.ultralytics.com/models/yoloe/), [Rockchip RKNN 공식 문서](https://docs.ultralytics.com/integrations/rockchip-rknn/). 검증 패키지는 `requirements.txt`에 기록했습니다.
