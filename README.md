# VAD 인수인계

카메라·영상에서 YOLO26 또는 YOLOE로 사람을 찾고, 조건이 일정 시간 지속되면 **이상 후보**를 표시하는 프로그램입니다. 이 브랜치에는 VAD 실행 코드만 있습니다.

## 시작하기

Python 3.11과 `uv`가 필요합니다. 이 브랜치를 받은 폴더에서 실행합니다.

```sh
git clone --branch feat/vad-handoff --single-branch https://github.com/PFSV/vision.git vision-vad
cd vision-vad
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r vad/requirements.txt
.venv/bin/python vad/detect.py --model yoloe-26n-seg.pt --source 0 \
  --prompts person "standing person" "person lying on the ground"
```

처음에는 모델 가중치를 내려받으므로 인터넷이 필요합니다. 기존 모델 파일이 있으면 `--model`에 경로를 넣습니다. macOS에서는 실행 앱의 카메라 권한을 허용해야 합니다. 화면에서 `q` 또는 Ctrl+C로 종료합니다.

영상 파일로 확인하려면:

```sh
.venv/bin/python vad/detect.py --model yoloe-26n-seg.pt --source sample.mp4 \
  --prompts person "standing person" "person lying on the ground" \
  --no-display --max-frames 100 --events /tmp/vad-events.jsonl
```

일반 YOLO26은 `--model yolo26n.pt`로 지정하고 `--prompts`를 빼면 됩니다. RKNN은 RK3588 런타임이 필요하며 Mac에서 검증하지 않았습니다.

## 어떤 조건을 잡는가

| 후보 | 기본 조건 |
|---|---|
| `possible_fall` | 사람 박스가 가로로 길거나, 누운 사람 프롬프트 검출 |
| `crowding` | 사람 관련 탐지 박스가 5개 초과 |
| `restricted_area` | 지정 구역에 사람 박스 중심이 진입. 기본 꺼짐 |

조건이 2초 지속돼야 후보로 표시합니다. 0.3초 이하 탐지 공백은 허용합니다. 객체 점수는 0.45 이상만 사용합니다. 후보 진입 시 콘솔과 선택한 JSONL 파일에 이벤트를 기록합니다.

`vad/config.json`에서 값을 바꿉니다. 제한 구역은 `restricted_region`을 `[0.2, 0.2, 0.8, 0.8]`처럼 가로·세로 비율 좌표로 지정합니다. 사람 박스 너비/높이 기준은 `wide_ratio`(기본 1.4), 인원 기준은 `max_people`입니다.

## 확인한 결과와 다음 작업

2026-10-08에 OpenCV 공개 샘플 `vtest.avi`의 100프레임(영상 10초)을 실제 처리하고 정상 종료했습니다. 첫 프레임은 768×576, 탐지 3개, 추론 약 236ms였습니다. 영상 9.9초에 `crowding` 후보 이벤트가 나왔습니다. 카메라는 macOS 권한 때문에 열리지 않았습니다.

이 결과는 실행 확인이며 정확도 평가는 아닙니다. 사람별 추적이 없어 중복 박스나 사람이 바뀌는 상황에 영향을 받습니다. 가로 박스만으로 실제 낙상을 확정할 수 없습니다. 실시간 처리가 느리면 탐지 공백으로 누적이 초기화될 수 있습니다.

인수 후에는 카메라 권한을 허용하고 실제 현장 영상에서 구역·점수·지속시간을 조절하세요. 정상/이상 영상과 비교해 오탐·미탐을 확인하는 것이 다음 작업입니다.

## 파일 안내

- `vad/detect.py`: 카메라·영상 입력, 모델 실행, 화면과 이벤트 출력
- `vad/rules.py`: 시간 누적과 이상 후보 규칙
- `vad/config.json`: 조절할 기준값
- `vad/requirements.txt`: 검증한 설치 버전

Bayesian 관련 기존 작업은 YOLO/RKNN 실행과 영상 전송, 설계 메모까지였습니다. Bayesian 확률 계산은 구현하지 않았습니다. 현재 객체 confidence를 이상 확률로 해석하지 않습니다.
