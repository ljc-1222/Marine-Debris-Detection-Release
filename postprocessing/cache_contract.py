"""Semantic identities for newly produced, independently sealed classifier caches."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/postprocess.yaml'
ROW_ID_ALGORITHM = 'sha256_candidate_crop_image_model_v1'
CODE_FILES = ('inference/classifiers.py', 'training/final_models.py', 'training/models.py',
              'training/hplus_lora.py', 'inference/detector_geometry.py', 'postprocessing/cache_contract.py')
EMPTY_EXECUTION = {'status': 'skipped_no_candidates', 'classifier_loaded': False,
                   'deployment_performed': False, 'forward_crops': 0}


def fingerprint(path: Path) -> dict[str, Any]:
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            checksum.update(block)
    return {'sha256': checksum.hexdigest(), 'bytes': Path(path).stat().st_size}


def class_order() -> list[str]:
    rows = json.loads((ROOT / 'configs/categories.json').read_text(encoding='utf-8'))['categories']
    if [row['id'] for row in rows] != list(range(1, 35)):
        raise ValueError('Expected the ordered project categories 1..34')
    return [row['name'] for row in rows]


def crop_rectangle(box_xywh: Sequence[float], context_scale: float) -> tuple[int, int, int, int]:
    x, y, width, height = map(float, box_xywh)
    right, bottom = x + width, y + height
    center_x, center_y = (x + right) / 2.0, (y + bottom) / 2.0
    half_width = (right - x) * context_scale / 2.0
    half_height = (bottom - y) * context_scale / 2.0
    left, top = math.floor(center_x - half_width), math.floor(center_y - half_height)
    right, bottom = math.ceil(center_x + half_width), math.ceil(center_y + half_height)
    return left, top, max(right, left + 1), max(bottom, top + 1)


def image_identities(index: Mapping[str, Any], images: Path, referenced: set[int]) -> dict[str, Any]:
    from inference.detector_geometry import encoded_image_info

    if not isinstance(index.get('images'), list) or not index['images']:
        raise ValueError('Index must contain images, even when there are no candidates')
    # An empty bank still binds and checks its real input images. No crop rows
    # are needed, but missing or replaced source files must not become a success.
    if not referenced:
        referenced = {int(row['id']) for row in index['images']}
    result, seen = {}, set()
    for row in index['images']:
        image_id = int(row['id'])
        if image_id in seen:
            raise ValueError('Index image IDs must be unique')
        seen.add(image_id)
        if image_id not in referenced:
            continue
        name = str(row['file_name'])
        relative = PurePosixPath(name)
        if not relative.parts or relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Index filenames must be safe relative image paths')
        path = images / name
        width, height = int(row['width']), int(row['height'])
        if width <= 0 or height <= 0 or encoded_image_info(path)[:2] != (width, height):
            raise ValueError('Encoded image dimensions disagree with the index: ' + name)
        result[str(image_id)] = {'file_name': name, 'width': width, 'height': height, **fingerprint(path)}
    if referenced != {int(key) for key in result}:
        raise ValueError('Candidate bank references unknown image IDs')
    return result


def row_ids(bank: Mapping[str, np.ndarray], model: str, config: Mapping[str, Any],
            images: Mapping[str, Any], checkpoint_sha256: str) -> np.ndarray:
    """Bind each output row to its full candidate, integer crop, source bytes and head."""
    count = len(bank['scores'])
    result = np.empty((count, 32), dtype=np.uint8)
    for index in range(count):
        image_id = int(bank['image_ids'][index])
        box = [float(value) for value in bank['boxes_xywh'][index]]
        value = {'row': index, 'ordinal': int(bank['ordinals'][index]), 'image_id': image_id,
                 'image': images[str(image_id)], 'category_id': int(bank['category_ids'][index]),
                 'box_xywh': box, 'score': float(bank['scores'][index]),
                 'crop_rectangle': crop_rectangle(box, float(config['context_scale'])),
                 'model': model, 'model_config': config, 'checkpoint_sha256': checkpoint_sha256}
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        result[index] = np.frombuffer(hashlib.sha256(encoded).digest(), dtype=np.uint8)
    return result


def checked_array_path(manifest: Path, descriptor: Mapping[str, Any]) -> Path:
    name = descriptor.get('path')
    relative = PurePosixPath(name) if isinstance(name, str) else PurePosixPath('/')
    if not relative.parts or relative.is_absolute() or '..' in relative.parts or '\\' in str(name):
        raise ValueError('New cache arrays must use safe relative paths')
    path = manifest.parent / name
    expected = {key: descriptor.get(key) for key in ('sha256', 'bytes')}
    if fingerprint(path) != expected:
        raise ValueError('Cache array SHA256/bytes mismatch: ' + str(path))
    return path


def verify_cache(manifest_path: Path, model: str, bank_path: Path, bank: Mapping[str, np.ndarray],
                 index_path: Path, images: Mapping[str, Any], checkpoint_manifest: Path,
                 expected_manifest_sha256: str, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Expected manifest SHA comes from the producer's trusted external receipt."""
    if fingerprint(manifest_path)['sha256'] != expected_manifest_sha256:
        raise ValueError('Sealed head manifest SHA256 mismatch: ' + model)
    document = json.loads(manifest_path.read_text(encoding='utf-8'))
    if document.get('schema_version') != 2 or document.get('model') != model:
        raise ValueError('Expected a version-2 cache for model ' + model)
    if (document.get('class_order') != class_order() or document.get('class_id_space') != 'foreground_0_based'
            or document.get('coordinate_system') != 'encoded_image_pixels'):
        raise ValueError('Cache class order or coordinate contract changed: ' + model)
    from inference.detector_geometry import ENCODED_DECODER
    if document.get('image_decode_policy') != ENCODED_DECODER:
        raise ValueError('Cache encoded-image decoding contract changed: ' + model)
    expected_inputs = {'bank': fingerprint(bank_path), 'index': fingerprint(index_path),
                       'config': fingerprint(config_path), 'checkpoint_catalog': fingerprint(checkpoint_manifest)}
    if document.get('inputs') != expected_inputs:
        raise ValueError('Cache bank/index/config/checkpoint catalog identity mismatch: ' + model)
    for field, key in (('candidate_sha256', 'bank'), ('index_sha256', 'index'),
                       ('config_sha256', 'config'), ('checkpoint_catalog_sha256', 'checkpoint_catalog')):
        if document.get(field) != expected_inputs[key]['sha256']:
            raise ValueError('Cache identity field mismatch: ' + field)
    from training.checkpoints import load_manifest
    catalog = load_manifest(checkpoint_manifest)
    artifact = catalog['artifacts'][model]
    if document.get('checkpoint_sha256') != artifact['sha256'] or document.get('checkpoint_bytes') != artifact['bytes']:
        raise ValueError('Cache checkpoint identity mismatch: ' + model)
    if 'classes' in catalog and catalog['classes'] != class_order():
        raise ValueError('Checkpoint catalog class order changed')
    if document.get('images') != images:
        raise ValueError('Cache source image identity mismatch: ' + model)
    settings = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    matches = [row for row in settings['dcr']['models'] if row['name'] == model]
    if len(matches) != 1 or document.get('model_config') != matches[0]:
        raise ValueError('Cache crop/model configuration mismatch: ' + model)
    if document.get('rows') != len(bank['scores']) or document.get('row_identity_algorithm') != ROW_ID_ALGORITHM:
        raise ValueError('Cache row identity contract changed: ' + model)
    if not len(bank['scores']) and (document.get('unique_crops') != 0 or document.get('execution') != EMPTY_EXECUTION):
        raise ValueError('Empty caches must record skipped classifier execution: ' + model)
    expected_deployment = 'merged_lora_160' if model == 'hplus' else 'full_model'
    if document.get('deployment') != expected_deployment:
        raise ValueError('Cache deployment contract changed: ' + model)
    if document.get('code') != {name: fingerprint(ROOT / name) for name in CODE_FILES}:
        raise ValueError('Cache producer code identity mismatch: ' + model)
    fields = ['class_logits', 'objectness_logits', 'row_ids'] + (['valid_crops'] if model == 'hplus' else [])
    paths = {name: checked_array_path(manifest_path, document[name]) for name in fields}
    actual_ids = np.load(paths['row_ids'], allow_pickle=False, mmap_mode='r')
    expected_ids = row_ids(bank, model, matches[0], images, artifact['sha256'])
    if actual_ids.dtype != np.uint8 or actual_ids.shape != expected_ids.shape or not np.array_equal(actual_ids, expected_ids):
        raise ValueError('Cache candidate/crop row order mismatch: ' + model)
    return document
