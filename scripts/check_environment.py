"""Check installed training dependencies, pinned sources, and CUDA operators."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement
if __package__:
    from .environment import source_revisions
else:
    from environment import source_revisions

ROOT = Path(__file__).resolve().parents[1]


def check_requirements(path: Path) -> dict[str, str]:
    installed = {}
    failures = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f'{requirement.name}: missing')
            continue
        installed[requirement.name] = version
        if not requirement.specifier.contains(version):
            failures.append(f'{requirement.name}: expected {requirement.specifier}, found {version}')
    if failures:
        raise RuntimeError('\n'.join(failures))
    return installed


def check_sources() -> dict:
    return source_revisions()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('training', 'codetr'), default='training')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Use Python 3.11 for the fixed training environment')
    dependencies = check_requirements(ROOT / ('requirements-codetr.txt' if args.mode == 'codetr' else 'requirements.txt'))
    sources = check_sources()
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16-capable CUDA GPU is required')
    if args.mode == 'codetr':
        sys.path.insert(0, str(ROOT))
        import training.codetr_runtime  # Check GEMM under the actual precision policy.
    value = torch.randn(8, 8, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = (value @ value.T).float().square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(value.grad).all():
        raise FloatingPointError('CUDA BF16 forward/backward check failed')
    if args.mode == 'codetr':
        from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
        value = torch.randn(1, 4, 1, 2, device='cuda', requires_grad=True)
        shapes = torch.tensor([[2, 2]], dtype=torch.long, device='cuda')
        starts = torch.tensor([0], dtype=torch.long, device='cuda')
        locations = torch.rand(1, 1, 1, 1, 1, 2, device='cuda')
        weights = torch.ones(1, 1, 1, 1, 1, device='cuda')
        result = MultiScaleDeformableAttnFunction.apply(value, shapes, starts, locations, weights, 64)
        result.sum().backward()
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise FloatingPointError('MMCV CUDA forward/backward check failed')
    print(json.dumps({'mode': args.mode, 'python': sys.version.split()[0],
                      'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name(), 'sources': sources,
                      'dependencies': dependencies, 'checks': 'CUDA forward/backward completed'}, indent=2))


if __name__ == '__main__':
    main()
