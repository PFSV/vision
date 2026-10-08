# 텍스트·비전 프롬프트 탐지

[문서 홈](../README.md) · [설치](installation.md)

YOLOE는 텍스트 클래스명 또는 참조 이미지의 객체 박스로 탐지 대상을 지정합니다. 프롬프트 지정은 추론 설정이며 모델 학습과는 별개입니다.

## 텍스트 프롬프트

```sh
python yolo/predict.py --model yoloe-26n-seg.pt --source 0 \
  --prompts person bottle "dog robot"
```

공백이 포함된 클래스명은 따옴표로 감쌉니다. 자세·상태를 표현한 프롬프트가 해당 의미를 정확하게 구분하는지는 실제 영상으로 검증해야 합니다.

## 비전 프롬프트

참조 이미지와 객체 박스를 JSON으로 지정합니다.

```sh
cp yolo/visual.example.json yolo/visual.json
```

`yolo/visual.json`을 수정합니다.

```json
{
  "image": "reference.jpg",
  "bboxes": [[10, 20, 100, 200]],
  "cls": [0],
  "names": ["person"]
}
```

| 필드 | 형식 | 의미 |
|---|---|---|
| `image` | 문자열 | 참조 이미지. 상대 경로는 JSON 파일 기준 |
| `bboxes` | 좌표 배열 | 참조 이미지의 픽셀 좌표 `[x1,y1,x2,y2]` |
| `cls` | 정수 배열 | 각 박스의 클래스 번호. 0부터 순서대로 지정 |
| `names` | 문자열 배열 | 클래스 번호에 대응하는 출력 이름 |

같은 클래스에 여러 박스를 지정할 수 있습니다. 박스 수와 `cls` 항목 수는 같아야 하며, 각 클래스에는 대응하는 이름이 필요합니다.

```sh
python yolo/predict.py --model yoloe-26n-seg.pt \
  --source sample.mp4 --visual yolo/visual.json
```

참조 박스에서 얻은 임베딩을 이후 프레임 탐지에 사용합니다. 이 도구는 YOLOE의 일반 `object0` 이름을 JSON의 `names`로 매핑합니다.

## 프롬프트 프로필

```sh
# 설정 저장
python yolo/predict.py --model yoloe-26n-seg.pt --source sample.mp4 \
  --prompts person bottle --save-embeddings prompts.npz \
  --no-display --max-frames 1

# 설정 재사용
python yolo/predict.py --model yoloe-26n-seg.pt --source sample.mp4 \
  --embeddings prompts.npz --no-display --max-frames 100
```

프로필은 모델 아키텍처와 연결됩니다. 다른 아키텍처의 모델에 사용할 수 없습니다. 동일한 프롬프트를 다른 가중치에 적용할 경우 결과를 다시 검증하십시오. 프로필은 원본 `.pt`에서 사용하며 RKNN 런타임 입력이 아닙니다.

## 실행 옵션 및 출력

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | `yoloe-26n-seg.pt` | 모델 파일 또는 정적 RKNN 폴더 |
| `--source` | `0` | 카메라 번호, 영상·이미지 파일, 스트림 주소 |
| `--imgsz` | `640` | 모델 입력 크기 |
| `--prompts` | 없음 | 텍스트 클래스 목록 |
| `--visual` | 없음 | 참조 이미지 JSON |
| `--embeddings` | 없음 | 저장된 NPZ 프로필 |
| `--save-embeddings` | 없음 | 프로필 저장 경로 |
| `--no-display` | 꺼짐 | 화면 없이 실행 |
| `--max-frames` | 제한 없음 | 처리할 최대 프레임 수 |

콘솔에는 프레임 번호(1부터 시작), 탐지 수, 단계별 처리 시간(ms)을 JSON으로 출력합니다. 객체 박스와 이름은 화면에 표시됩니다. 프롬프트 옵션 세 가지는 상호 배타적입니다. 원본 YOLOE `.pt`만 동적 프롬프트를 지원합니다.

API 참고: [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/).
