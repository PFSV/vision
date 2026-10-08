"""Shared text / reference-image prompting for YOLOE."""
import json
from pathlib import Path


def add_prompt_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prompts', nargs='+', help='YOLOE text class names')
    group.add_argument('--visual', type=Path, help='Reference image and boxes JSON')
    group.add_argument('--embeddings', type=Path, help='Previously saved YOLOE .npz prompt profile')


def configure(model, args):
    if args.prompts:
        model.set_classes(args.prompts)
    elif args.embeddings:
        model.load_prompt_embeddings(args.embeddings)
    elif args.visual:
        import numpy as np
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        data = json.loads(args.visual.read_text())
        boxes = np.asarray(data['bboxes'], dtype=float)
        classes = np.asarray(data['cls'], dtype=int)
        names = data['names']
        if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) != len(classes):
            raise ValueError('Need Nx4 pixel boxes and N class IDs')
        if not np.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError('Boxes must be finite [x1,y1,x2,y2] with positive area')
        if sorted(set(classes.tolist())) != list(range(len(names))):
            raise ValueError('Class IDs must be sequential from 0 and match names')
        reference = Path(data['image'])
        if not reference.is_absolute():
            reference = args.visual.parent / reference
        if not reference.is_file():
            raise FileNotFoundError(reference)
        model.predict(str(reference), refer_image=str(reference),
                      visual_prompts={'bboxes': boxes, 'cls': classes},
                      predictor=YOLOEVPSegPredictor, verbose=False)
        # Reference prompting creates generic object0 names; preserve embeddings, map names.
        model.set_classes(names, model.model.pe)
        model.predictor = None


def load_model(args):
    from ultralytics import YOLO, YOLOE
    prompted = args.prompts or args.visual or args.embeddings
    if prompted and Path(args.model).suffix != '.pt':
        raise ValueError('Dynamic prompts require original YOLOE .pt, not an RKNN export')
    model = YOLOE(args.model) if prompted else YOLO(args.model)
    configure(model, args)
    return model
