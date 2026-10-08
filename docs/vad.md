# VAD 설정 및 이벤트

[문서 홈](../README.md) · [탐지](detection.md)

VAD는 Visual Anomaly Detection을 의미합니다. 객체 탐지 결과에 장면 단위 규칙을 적용하여 이상 후보를 출력합니다.

## 실행

```sh
python yolo/vad/detect.py --model yoloe-26n-seg.pt --source 0 \
  --prompts person "standing person" "person lying on the ground"
```

영상 파일을 화면 없이 처리하고 이벤트를 저장하려면:

```sh
python yolo/vad/detect.py --model yoloe-26n-seg.pt --source sample.mp4 \
  --prompts person "standing person" "person lying on the ground" \
  --no-display --events /tmp/vad-events.jsonl --max-frames 100
```

`--config`로 별도 설정 JSON을 지정할 수 있습니다. 기본 설정은 `yolo/vad/config.json`입니다. `--visual`과 `--embeddings`도 지원하며, 모델의 사람 클래스명은 설정의 `person_labels`와 일치해야 합니다.

## 후보 판정

| 이벤트 | 조건 |
|---|---|
| `possible_fall` | 사람 박스의 너비/높이가 기준 이상이거나 누운 사람 클래스 검출 |
| `crowding` | 사람 관련 박스 수가 기준 인원을 초과 |
| `restricted_area` | 사람 박스 중심이 지정한 사각 구역 내부 |

조건을 `hold_seconds` 동안 누적한 뒤 후보로 표시합니다. 조건이 사라지면 활성 표시가 해제됩니다. `gap_seconds` 이내의 짧은 탐지 공백은 누적을 유지하고, 더 긴 공백은 누적을 초기화합니다. 이벤트는 활성 상태에 진입할 때 기록합니다.

영상 파일은 프레임 수/FPS 기반의 영상 시간을 사용합니다. 카메라·스트림은 실제 경과 시간을 사용합니다. 관측되지 않은 프레임의 상태는 추정하지 않습니다.

## 설정 참조

| 키 | 기본값 | 의미 |
|---|---|---|
| `confidence` | `0.45` | 사용할 최소 객체 탐지 점수 |
| `hold_seconds` | `2.0` | 후보 판정까지 누적할 시간(초) |
| `gap_seconds` | `0.3` | 허용할 탐지 공백(초) |
| `wide_ratio` | `1.4` | 낙상 의심에 사용할 박스 너비/높이 |
| `max_people` | `5` | 허용 인원. 초과할 때 군중 후보 |
| `restricted_region` | `null` | 제한 구역. `null`은 비활성화 |
| `person_labels` | 설정 파일 참고 | 사람으로 집계할 클래스 이름 |
| `lying_labels` | 설정 파일 참고 | 누운 사람으로 판단할 클래스 이름 |

`confidence`는 객체 탐지 점수이며 이상일 확률이 아닙니다. 제한 구역은 `[x1,y1,x2,y2]` 형식의 0~1 비율 좌표입니다. 예를 들어 `[0.2,0.2,0.8,0.8]`은 중앙 사각 영역을 의미합니다.

## 이벤트 형식

이벤트 파일은 append 방식의 JSONL입니다. 각 줄은 하나의 이벤트입니다. 아래 값은 필드 설명을 위한 예시입니다.

```json
{"rule":"crowding","frame":99,"source_seconds":9.9,"unix_time":1791443714.0,"status":"candidate"}
```

| 필드 | 의미 |
|---|---|
| `rule` | 후보 규칙 이름 |
| `frame` | 0부터 시작하는 영상 프레임 번호 |
| `source_seconds` | 파일 영상 내 위치(초). 실시간 입력은 `null` |
| `unix_time` | 이벤트 처리 시각의 Unix timestamp |
| `status` | 현재 `candidate` |

콘솔에는 첫 추론 완료 정보(`inference_ready`)와 종료 시 처리 프레임 수(`stopped`)도 출력됩니다. 이 두 상태 메시지는 이벤트 파일에 기록하지 않습니다.

## 적용 범위

사람별 추적을 수행하지 않으므로 서로 다른 사람이 조건을 이어도 같은 장면 후보로 누적될 수 있습니다. 박스 중복과 객체 미탐에 영향을 받습니다. 가로로 긴 박스는 자세의 근사치이며 실제 낙상을 확정하지 않습니다. 현장 영상으로 점수·시간·구역 기준을 조정해야 합니다.

Bayesian 추론, 학습된 행동 분류, 이상 확률 계산은 현재 구현에 포함하지 않습니다.
