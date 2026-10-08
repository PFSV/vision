# Vision

YOLO26 및 YOLOE를 이용한 객체 탐지, 모델 학습, Rockchip NPU 배포, 영상 이상 후보 감지 도구입니다.

## 기능

| 기능 | 설명 | 안내 |
|---|---|---|
| Rockchip 배포 | YOLO26 및 프롬프트를 고정한 YOLOE를 RKNN으로 변환 | [배포 가이드](docs/rockchip.md) |
| 오픈 보캐뷸러리 탐지 | 텍스트 또는 참조 이미지로 탐지 대상 지정 | [탐지 가이드](docs/detection.md) |
| 모델 학습 | YOLO26 학습, YOLOE 미세 조정 및 텍스트·비전 프롬프트 학습 | [학습 가이드](docs/training.md) |
| VAD | 탐지 결과에 시간 누적 규칙을 적용해 이상 후보 출력 | [VAD 가이드](docs/vad.md) |

## 빠른 시작

Python 3.11, Git, `uv`가 필요합니다. 아래 명령은 macOS 또는 Linux 셸 기준입니다.

```sh
git clone --branch feat/vision-pipeline --single-branch https://github.com/PFSV/vision.git
cd vision
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r yolo/requirements.txt
source .venv/bin/activate

python yolo/predict.py --model yoloe-26n-seg.pt --source 0 --prompts person bottle
```

카메라 대신 영상 파일을 사용하려면 `--source sample.mp4`를 지정합니다. 첫 실행에는 모델과 텍스트 인코더 다운로드가 필요합니다. 화면에서 `q` 또는 터미널에서 Ctrl+C를 누르면 종료합니다.

## 문서

- [설치 및 실행 환경](docs/installation.md)
- [Rockchip 모델 변환 및 배포](docs/rockchip.md)
- [텍스트·비전 프롬프트 탐지](docs/detection.md)
- [데이터 준비 및 학습](docs/training.md)
- [VAD 설정 및 이벤트 형식](docs/vad.md)
- [문제 해결](docs/troubleshooting.md)
- [검증 범위](yolo/VALIDATION.md)

## 검증 상태

PyTorch 탐지, 프롬프트 재사용, ONNX 변환, 소형 공개 데이터 학습 및 VAD 실행을 확인했습니다. RKNN 변환과 Rockchip 보드 실행, 현장 탐지 정확도는 아직 검증하지 않았습니다.

## 디렉터리

```text
docs/                         설치·사용·배포 안내
yolo/predict.py               객체 탐지
yolo/prompting.py             공통 프롬프트 처리
yolo/train.py                 모델 학습
yolo/yolo26n-rknn/install.py  모델 변환
yolo/vad/                     이상 후보 감지 및 설정
yolo/tests/                   규칙 검증
yolo/requirements*.txt        환경별 의존성
```
