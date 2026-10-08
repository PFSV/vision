"""Extend the original YOLO26 -> RKNN export with prompted YOLOE support."""
import argparse
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompting import add_prompt_args, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='yolo26n.pt')
    p.add_argument('--target', default='rk3588')
    p.add_argument('--quantize', type=int, choices=[8, 16], default=8)
    p.add_argument('--data', default='coco8.yaml', help='YOLO dataset YAML for INT8 calibration')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--format', choices=['rknn', 'onnx'], default='rknn')
    add_prompt_args(p)
    args = p.parse_args()
    if args.format == 'rknn' and (platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64')):
        p.error('RKNN conversion needs x86 Linux + rknn-toolkit2. On Mac use --format onnx, then convert on Linux.')
    model = load_model(args)
    if args.format == 'rknn' and args.quantize == 8 and model.task != 'detect':
        p.error('RKNN INT8 supports detection only. Use --quantize 16 for a YOLOE segmentation model.')
    options = dict(format=args.format, imgsz=args.imgsz, opset=19)
    if args.format == 'rknn':
        options.update(name=args.target, quantize=args.quantize)
        if args.quantize == 8:
            options['data'] = args.data
    print(model.export(**options))


if __name__ == '__main__':
    main()
