"""Temporal rules for visual anomaly candidates; no Bayesian inference."""
import math


class RuleDetector:
    def __init__(self, config):
        self.config = config
        self.started = {}
        self.last_seen = {}
        self.active = set()
        for key in ('hold_seconds', 'gap_seconds', 'confidence', 'wide_ratio'):
            value = config[key]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid {key}')
        if config['max_people'] < 0:
            raise ValueError('max_people must be nonnegative')
        region = config.get('restricted_region')
        if region is not None and not (
            len(region) == 4 and 0 <= region[0] < region[2] <= 1
            and 0 <= region[1] < region[3] <= 1
        ):
            raise ValueError('restricted_region must be normalized [x1,y1,x2,y2]')

    def update(self, detections, now, width, height):
        people = [d for d in detections if d['confidence'] >= self.config['confidence']
                  and d['label'].lower() in self.config['person_labels']]
        candidates = set()
        if len(people) > self.config['max_people']:
            candidates.add('crowding')
        for detection in people:
            x1, y1, x2, y2 = detection['box']
            if (x2 - x1) / max(y2 - y1, 1) >= self.config['wide_ratio']:
                candidates.add('possible_fall')
            if detection['label'].lower() in self.config['lying_labels']:
                candidates.add('possible_fall')
            region = self.config.get('restricted_region')
            if region:
                cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                if region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
                    candidates.add('restricted_area')
        for rule in list(self.started):
            if now - self.last_seen[rule] > self.config['gap_seconds']:
                del self.started[rule]
                del self.last_seen[rule]
        for rule in candidates:
            self.started.setdefault(rule, now)
            self.last_seen[rule] = now
        active = {rule for rule in candidates
                  if now - self.started[rule] >= self.config['hold_seconds']}
        entered = active - self.active
        self.active = active
        return sorted(active), sorted(entered)
