# 설치 및 실행 환경

[문서 홈](../README.md)

## 실행 환경

| 작업 | 환경 | 추가 조건 |
|---|---|---|
| PyTorch 탐지 및 VAD | macOS 또는 Linux, Python 3.11 | 카메라 또는 영상 파일 |
| 모델 학습 | CPU, Apple MPS 또는 CUDA | 이미지와 정답 라벨 |
| RKNN 변환 | x86 Linux | RKNN Toolkit2 |
| RKNN 추론 | Rockchip Linux 보드 | NPU 드라이버 및 호환 런타임 |

Mac에서는 ONNX 변환까지 확인할 수 있습니다. RKNN 변환은 x86 Linux에서 실행합니다. 하드웨어 검증 범위는 [검증 기록](../yolo/VALIDATION.md)을 참고하십시오.

## 기본 설치

```sh
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r yolo/requirements.txt
source .venv/bin/activate
```

모든 문서의 명령은 저장소 루트에서 가상환경을 활성화한 상태를 기준으로 합니다. `uv` 설치 안내: [공식 설치 문서](https://docs.astral.sh/uv/getting-started/installation/).

## 변환 환경

```sh
uv pip install --python .venv/bin/python -r yolo/requirements-export.txt
```

이 파일은 ONNX 도구를 설치하고, x86 Linux에서 RKNN Toolkit2를 추가합니다.

## Rockchip 실행 환경

보드에 저장소와 변환된 모델 폴더를 복사한 뒤 Python 3.11 가상환경을 준비합니다.

```sh
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r yolo/requirements-rockchip.txt
source .venv/bin/activate
```

Python 패키지 설치와 별도로 보드의 NPU 드라이버 및 런타임이 호환되어야 합니다. 이 저장소는 OS 이미지 또는 드라이버를 설치하지 않습니다.

## 모델 및 입력 준비

기본 모델명을 지정하면 Ultralytics가 필요한 가중치를 다운로드합니다. 오프라인 환경에서는 모델 경로를 직접 지정합니다. YOLOE 텍스트 프롬프트는 최초 사용 시 추가 텍스트 인코더를 다운로드합니다. 실행할 디렉터리에서 한 번 초기화하거나 [프롬프트 프로필](detection.md#프롬프트-프로필)을 준비합니다.

`--source 0`은 카메라 인덱스입니다. 장치 번호는 환경마다 다를 수 있습니다. macOS에서는 시스템 설정 → 개인정보 보호 및 보안 → 카메라에서 실행 앱을 허용합니다.

화면 출력이 없는 환경에서는 `--no-display`를 지정합니다. 유한한 실행 확인에는 `--max-frames 100`을 사용할 수 있습니다.
