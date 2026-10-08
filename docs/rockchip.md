# Rockchip 모델 변환 및 배포

[문서 홈](../README.md) · [설치](installation.md)

## 배포 흐름

1. 탐지할 클래스 또는 학습된 모델을 준비합니다.
2. x86 Linux에서 RKNN 모델로 변환합니다.
3. 변환 폴더를 Rockchip 보드에 복사합니다.
4. 보드에서 추론하고 현장 영상으로 결과를 검증합니다.

RKNN으로 변환한 YOLOE는 탐지 클래스가 고정됩니다. 새로운 텍스트·이미지 프롬프트를 적용하려면 원본 `.pt`에서 다시 변환합니다.

## YOLO26 INT8 변환

기존 `yolo/yolo26n-rknn/install.py`를 사용합니다.

```sh
python yolo/yolo26n-rknn/install.py \
  --model yolo26n.pt --target rk3588 --quantize 8 --data coco8.yaml
```

`coco8.yaml`은 실행 확인용 예시입니다. 실제 배포에서는 현장 이미지를 담은 데이터셋 YAML을 `--data`로 지정하여 INT8 보정을 수행합니다.

## YOLOE FP16 변환

```sh
python yolo/yolo26n-rknn/install.py \
  --model yoloe-26n-seg.pt --target rk3588 --quantize 16 \
  --prompts person bottle
```

YOLOE segmentation 모델에는 `--quantize 16`을 사용합니다. 현재 변환 진입점은 INT8을 detection 모델로 제한합니다. 이미지 프롬프트는 `--visual yolo/visual.json`, 저장된 프로필은 `--embeddings prompts.npz`로 지정할 수 있습니다. 세 프롬프트 옵션은 함께 사용할 수 없습니다.

## 변환 결과

변환 결과는 모델 옆의 `*_rknn_model/` 디렉터리에 생성됩니다.

```text
yolo26n_rknn_model/
  yolo26n-rk3588.rknn
  metadata.yaml
  dataset.txt            INT8 보정에 사용한 이미지 목록
```

`.rknn`과 `metadata.yaml`을 포함한 폴더를 함께 보드에 복사합니다. 기존 변환 파일은 `yolo/yolo26n-rknn/yolo26n_rknn_model/`에 있습니다. 해당 파일의 존재는 현재 보드에서의 호환성을 보장하지 않습니다.

## 보드에서 실행

```sh
python yolo/predict.py --model ./yolo26n_rknn_model --source 0
```

VAD를 연결하려면:

```sh
python yolo/vad/detect.py --model ./yolo26n_rknn_model --source 0
```

정적 RKNN 모델을 사용할 때는 `--prompts`, `--visual`, `--embeddings`를 지정하지 않습니다. VAD 설정의 사람 클래스명은 모델의 클래스명과 맞아야 합니다.

## Mac에서 ONNX 내보내기

[변환 의존성](installation.md#변환-환경)을 설치한 뒤 실행합니다.

```sh
python yolo/yolo26n-rknn/install.py \
  --model yoloe-26n-seg.pt --format onnx --prompts person bottle
```

모델 옆에 `.onnx` 파일이 생성됩니다. 이 명령은 ONNX 내보내기 확인용입니다. RKNN을 만들려면 x86 Linux에 원본 `.pt`와 프롬프트 설정을 준비해 RKNN 변환 명령을 실행합니다. 현재 스크립트는 ONNX 파일을 입력받아 RKNN으로 변환하는 별도 진입점을 제공하지 않습니다.

## 변환 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | `yolo26n.pt` | 원본 모델 |
| `--format` | `rknn` | `rknn` 또는 `onnx` |
| `--target` | `rk3588` | Rockchip 대상 칩 |
| `--quantize` | `8` | RKNN INT8 또는 FP16(`16`) |
| `--data` | `coco8.yaml` | INT8 보정 데이터 YAML |
| `--imgsz` | `640` | 모델 입력 크기 |

API 참고: [Ultralytics Rockchip RKNN](https://docs.ultralytics.com/integrations/rockchip-rknn/).
