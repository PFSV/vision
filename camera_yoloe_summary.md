# Logitech 카메라와 YOLOE-26 Nano 사용 정리

## 1. OpenCV 카메라 번호 확인

OpenCV의 카메라 번호는 카메라 이름이 아니라 실행 환경에서 부여되는 인덱스입니다. USB Logitech 카메라라고 해서 반드시 `1`번인 것은 아닙니다.

```python
import cv2

for number in range(10):
    camera = cv2.VideoCapture(number)

    if camera.isOpened():
        success, frame = camera.read()
        if success:
            print(number, "사용 가능")

    camera.release()
```

`사용 가능`으로 출력된 번호를 하나씩 화면으로 확인하여 Logitech 카메라의 번호를 찾습니다.

macOS에서는 터미널에서 장치 목록을 확인할 수도 있습니다.

```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

## 2. 카메라에서 한 장의 이미지 읽기

찾은 번호를 `camera_index`에 입력합니다.

```python
import cv2
import matplotlib.pyplot as plt

camera_index = 0  # Logitech 카메라 번호로 변경
camera = cv2.VideoCapture(camera_index)

success, frame = camera.read()

if success:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.axis("off")
    plt.show()
else:
    print("카메라 프레임을 읽을 수 없습니다.")

camera.release()
```

`frame.shape`는 프레임의 높이, 너비, 색상 채널을 확인합니다.

```python
frame.shape
```

## 3. Jupyter Notebook에서 실시간 카메라 화면 보기

Jupyter에서는 `cv2.imshow()` 대신 Matplotlib 화면을 갱신하는 방법을 사용할 수 있습니다.

```python
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

camera_index = 0  # Logitech 카메라 번호로 변경
camera = cv2.VideoCapture(camera_index)

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        clear_output(wait=True)
        plt.figure(figsize=(10, 6))
        plt.imshow(frame_rgb)
        plt.axis("off")
        display(plt.gcf())
        plt.close()

except KeyboardInterrupt:
    print("카메라를 종료했습니다.")

finally:
    camera.release()
```

실행 중인 셀의 중지 버튼을 누르거나 `KeyboardInterrupt`로 종료합니다.

## 4. YOLOE-26 Nano 모델 불러오기

작업 폴더에 `yoloe-26n-seg.pt` 파일이 있어야 합니다.

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26n-seg.pt")
```

## 5. 텍스트 프롬프트로 탐지할 객체 지정

`set_classes()`에 탐지할 객체 이름을 입력합니다. 짧은 영어 객체명을 사용하는 것이 안정적입니다.

```python
model.set_classes([
    "person",
    "cup",
    "keyboard",
    "mouse",
    "bottle"
])
```

한 종류만 탐지하려면 다음처럼 사용합니다.

```python
model.set_classes(["person"])
```

## 6. 실시간 카메라 + YOLOE 탐지

```python
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

camera_index = 0  # Logitech 카메라 번호로 변경
camera = cv2.VideoCapture(camera_index)

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        results = model.predict(source=frame, verbose=False)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        clear_output(wait=True)
        plt.figure(figsize=(10, 6))
        plt.imshow(annotated_frame)
        plt.axis("off")
        display(plt.gcf())
        plt.close()

except KeyboardInterrupt:
    print("탐지를 종료했습니다.")

finally:
    camera.release()
```

## 7. 핵심 확인 사항

- `camera_index`는 Logitech 화면이 실제로 나오는 번호로 설정합니다.
- `0`번은 보통 MacBook 내장 카메라이지만, 환경에 따라 달라질 수 있습니다.
- 카메라 번호는 고정되지 않을 수 있으므로 필요할 때 다시 검색합니다.
- 노트북에서 실시간 화면이 느리면 Matplotlib 갱신 대신 별도의 Python 스크립트와 `cv2.imshow()`를 사용하는 것이 더 빠릅니다.
- macOS에서 카메라 권한이 필요하면 VS Code 또는 사용 중인 Python 앱에 카메라 접근 권한을 허용합니다.
