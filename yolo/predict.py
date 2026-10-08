"""Text / visual open-vocabulary detection; also loads static RKNN directories."""
import argparse
import json
from pathlib import Path
from prompting import add_prompt_args, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='yoloe-26n-seg.pt')
    p.add_argument('--source', default='0')
    add_prompt_args(p)
    p.add_argument('--save-embeddings', type=Path)
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--max-frames', type=int)
    p.add_argument('--imgsz', type=int, default=640)
    args = p.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        p.error('--max-frames must be positive')
    model = load_model(args)
    if args.save_embeddings:
        if not (args.prompts or args.visual or args.embeddings):
            p.error('--save-embeddings requires prompts')
        model.save_prompt_embeddings(args.save_embeddings)
    source = int(args.source) if args.source.isdigit() else args.source
    results = model.predict(source=source, stream=True, imgsz=args.imgsz, verbose=False)
    count = 0
    try:
        for result in results:
            count += 1
            print(json.dumps({'frame': count, 'detections': len(result.boxes) if result.boxes is not None else 0,
                              'speed_ms': result.speed}), flush=True)
            if not args.no_display:
                import cv2
                cv2.imshow('YOLO detection', result.plot())
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            if args.max_frames and count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        results.close()
        predictor = model.predictor
        dataset = getattr(predictor, 'dataset', None)
        for cap in getattr(dataset, 'caps', []) or []:
            cap.release()
        cap = getattr(dataset, 'cap', None)
        if cap is not None:
            cap.release()
        if not args.no_display:
            import cv2
            cv2.destroyAllWindows()
    print(json.dumps({'processed_frames': count}), flush=True)


if __name__ == '__main__':
    main()
