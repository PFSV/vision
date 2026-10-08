"""YOLO fine-tuning and YOLOE text / visual-prompt training entrypoint."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['yolo', 'finetune', 'text', 'visual'])
    p.add_argument('--model', required=True, help='Pretrained .pt (visual needs an unfused text-prompt checkpoint)')
    p.add_argument('--data', required=True, help='YOLO dataset YAML, or multimodal train/val recipe YAML')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu', help='cpu, mps, or CUDA device number')
    p.add_argument('--project', default='runs/vision')
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--warmup-epochs', type=float, default=3.0)
    p.add_argument('--nbs', type=int, default=64, help='Nominal batch size used for gradient accumulation')
    args = p.parse_args()
    if args.epochs <= 0 or args.batch <= 0 or args.nbs <= 0 or not 0 < args.fraction <= 1:
        p.error('epochs/batch must be positive and fraction in (0,1]')
    from ultralytics import YOLO, YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer, YOLOESegTrainerFromScratch, YOLOESegVPTrainer
    import yaml
    model = YOLO(args.model) if args.mode == 'yolo' else YOLOE(args.model)
    options = dict(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                   device=args.device, project=args.project, name=args.mode,
                   workers=0, compile=False, fraction=args.fraction)
    options['warmup_epochs'] = args.warmup_epochs
    options['nbs'] = args.nbs
    if args.mode == 'finetune':
        options['trainer'] = YOLOEPESegTrainer
    elif args.mode in ('text', 'visual'):
        if model.task != 'segment':
            p.error('Text/visual recipe expects a YOLOE segmentation checkpoint')
        if getattr(model.model.model[-1], 'is_fused', False):
            p.error('Text/visual requires an unfused checkpoint, not fixed-class fine-tuning output')
        recipe = yaml.safe_load(Path(args.data).read_text())
        if isinstance(recipe.get('train'), dict):
            options['data'] = recipe
        else:
            dataset = str(Path(args.data).resolve())
            options['data'] = {'train': {'yolo_data': [dataset]}, 'val': {'yolo_data': [dataset]}}
        options['trainer'] = YOLOESegTrainerFromScratch if args.mode == 'text' else YOLOESegVPTrainer
        if args.mode == 'visual':
            head_index = len(model.model.model) - 1
            head = model.model.model[-1]
            if not hasattr(head, 'savpe'):
                p.error('Visual training needs SAVPE in an unfused text-prompt checkpoint')
            options['freeze'] = list(range(head_index)) + [
                f'{head_index}.{name}' for name, _ in head.named_children() if name != 'savpe']
    model.train(**options)
    print('Best weights:', model.trainer.best)


if __name__ == '__main__':
    main()
