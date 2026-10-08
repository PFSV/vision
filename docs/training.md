# 데이터 준비 및 학습

[문서 홈](../README.md) · [설치](installation.md)

## 학습 모드 선택

| `--mode` | 목적 | 입력 라벨 | 특성 |
|---|---|---|---|
| `yolo` | 일반 YOLO26 학습 | bounding box | 학습 클래스 탐지 |
| `finetune` | YOLOE 고정 클래스 미세 조정 | polygon | 클래스명을 모델에 고정 |
| `text` | YOLOE 텍스트·이미지 학습 | polygon + 클래스명 | 텍스트 프롬프트 경로 학습 |
| `visual` | YOLOE 비전 프롬프트 학습 | polygon | SAVPE 인코더 학습 |

텍스트만 입력해서 모델을 학습할 수는 없습니다. 클래스명이 있는 이미지와 정답 라벨이 필요합니다. 참조 이미지 한 장을 지정하는 [비전 프롬프트 탐지](detection.md)는 학습이 아닙니다.

## 데이터셋 구성

영상은 이미지 프레임으로 준비합니다. 학습·검증 세트는 서로 다른 영상이나 촬영 구간으로 나누어 인접 프레임이 섞이지 않도록 합니다.

```text
vision-data/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

이미지와 라벨은 같은 파일명을 사용합니다. 예를 들어 `images/train/frame001.jpg`의 라벨은 `labels/train/frame001.txt`입니다.

- detection 라벨 한 줄: `class_id center_x center_y width height`
- segmentation 라벨 한 줄: `class_id x1 y1 x2 y2 ...`

좌표는 이미지 크기로 나눈 0~1 값입니다. segmentation 라벨은 객체 윤곽을 표현하는 polygon 좌표입니다.

`yolo/dataset.example.yaml`을 복사해 데이터 경로와 클래스명을 수정합니다.

```yaml
path: /absolute/path/to/vision-data
train: images/train
val: images/val
names:
  0: standing person
  1: sitting person
  2: person lying on the ground
```

`path`는 데이터 루트이며 `train`과 `val`은 그 아래의 상대 경로입니다. 클래스 번호는 라벨의 번호와 일치해야 합니다. 이 YAML은 데이터를 생성하거나 자동 라벨링하지 않습니다.

## 학습 실행

### YOLO26

```sh
python yolo/train.py --mode yolo --model yolo26n.pt \
  --data dataset.yaml --device 0
```

### YOLOE 고정 클래스 미세 조정

```sh
python yolo/train.py --mode finetune --model yoloe-26n-seg.pt \
  --data dataset-seg.yaml --device 0
```

이 모드는 클래스명을 모델 헤드에 고정합니다. 이후 텍스트·비전 프롬프트 학습의 입력으로 사용하지 않습니다.

### YOLOE 텍스트·이미지 학습

```sh
python yolo/train.py --mode text --model yoloe-26n-seg.pt \
  --data dataset-seg.yaml --device 0
```

### YOLOE 비전 프롬프트 학습

```sh
python yolo/train.py --mode visual \
  --model runs/vision/text/weights/best.pt \
  --data dataset-seg.yaml --device 0
```

텍스트 학습 결과 또는 원본의 unfused YOLOE segmentation 체크포인트를 사용합니다. 나머지 레이어를 고정하고 SAVPE(비전 프롬프트 인코더)를 학습합니다. 정적 export와 `finetune` 결과는 사용하지 않습니다.

## 학습 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--mode` | 필수 | `yolo`, `finetune`, `text`, `visual` |
| `--model` | 필수 | 초기 모델 또는 체크포인트 |
| `--data` | 필수 | 데이터셋 YAML |
| `--epochs` | `30` | 학습 에폭 수 |
| `--batch` | `8` | 배치 크기 |
| `--imgsz` | `640` | 입력 크기 |
| `--device` | `cpu` | CPU, `mps`, CUDA 번호(`0`) |
| `--project` | `runs/vision` | 결과 루트 |
| `--fraction` | `1.0` | 학습 데이터 사용 비율 |
| `--warmup-epochs` | `3.0` | 워밍업 에폭 수 |
| `--nbs` | `64` | gradient accumulation의 기준 배치 크기 |

CPU 경로는 검증했습니다. MPS와 CUDA는 지정 가능한 옵션이며 이 프로젝트의 실행 검증 환경은 아닙니다.

## 결과 및 배포

결과는 `runs/vision/<mode>/` 아래에 생성됩니다. 같은 모드를 반복하면 디렉터리 이름에 번호가 붙습니다. 실행 마지막의 `Best weights:` 경로를 확인하십시오.

- `weights/best.pt`: 검증 기준으로 선택한 체크포인트
- `weights/last.pt`: 마지막 에폭 체크포인트
- `results.csv`: 학습·검증 지표

정답 라벨이 있는 별도 영상·이미지로 오탐과 미탐을 확인한 뒤 [Rockchip 배포](rockchip.md)로 진행합니다. 학습 데이터, 체크포인트, 실행 로그는 저장소의 커밋 대상에서 제외합니다.

## Grounding 데이터

`text`와 `visual`은 YOLO 데이터와 문장-영역 grounding 데이터를 결합한 YAML도 받을 수 있습니다.

```yaml
train:
  yolo_data:
    - /absolute/path/to/dataset-seg.yaml
  grounding_data:
    - img_path: /absolute/path/to/grounding-images
      json_file: /absolute/path/to/grounding-segments.json
val:
  yolo_data:
    - /absolute/path/to/validation-seg.yaml
```

이 형식은 데이터 준비가 별도로 필요하며 현재 검증은 일반 YOLO segmentation 데이터로 수행했습니다. 참고: [YOLOE 학습 문서](https://docs.ultralytics.com/models/yoloe/).
