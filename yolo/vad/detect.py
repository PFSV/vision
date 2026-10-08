"""YOLO26 / YOLOE camera or video detection with temporal anomaly rules."""
import argparse
import json
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompting import add_prompt_args, load_model
from rules import RuleDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', default='0', help='Camera index, video file, or stream URL')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    add_prompt_args(parser)
    parser.add_argument('--events', type=Path, help='Optional JSONL event output')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--max-frames', type=int, help='Stop after this many frames for a bounded run')
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    import cv2

    config = json.loads(args.config.read_text())
    detector = RuleDetector(config)
    model = load_model(args)
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f'Cannot open source: {args.source}')
    is_file = isinstance(source, str) and Path(source).is_file()
    fps = capture.get(cv2.CAP_PROP_FPS)
    if is_file and not (fps > 0):
        capture.release()
        raise RuntimeError('Video FPS unavailable; temporal rules need a valid FPS')
    event_file = None
    frame_number = 0
    try:
        if args.events:
            event_file = args.events.open('a', encoding='utf-8')
        while True:
            success, frame = capture.read()
            if not success:
                break
            # File time follows the source timeline, not inference throughput.
            now = frame_number / fps if is_file else time.monotonic()
            result = model.predict(frame, conf=config['confidence'], verbose=False)[0]
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    detections.append({'label': result.names[int(box.cls.item())],
                                       'confidence': float(box.conf.item()),
                                       'box': box.xyxy[0].cpu().tolist()})
            height, width = frame.shape[:2]
            if frame_number == 0:
                print(json.dumps({'status': 'inference_ready', 'width': width, 'height': height,
                                  'detections': len(detections), 'speed_ms': result.speed}), flush=True)
            active, entered = detector.update(detections, now, width, height)
            for rule in entered:
                event = {'rule': rule, 'frame': frame_number, 'source_seconds': now if is_file else None,
                         'unix_time': time.time(), 'status': 'candidate'}
                line = json.dumps(event)
                print(line, flush=True)
                if event_file:
                    event_file.write(line + '\n')
                    event_file.flush()
            if not args.no_display:
                annotated = result.plot()
                region = config.get('restricted_region')
                if region:
                    x1, y1, x2, y2 = region
                    cv2.rectangle(annotated, (int(x1 * width), int(y1 * height)),
                                  (int(x2 * width), int(y2 * height)), (0, 200, 255), 2)
                cv2.putText(annotated, ', '.join(active) or 'No anomaly candidate',
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255) if active else (0, 200, 0), 2)
                cv2.imshow('Visual anomaly candidates', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_number += 1
            if args.max_frames and frame_number >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if event_file:
            event_file.close()
        print(json.dumps({'status': 'stopped', 'processed_frames': frame_number}), flush=True)


if __name__ == '__main__':
    main()
