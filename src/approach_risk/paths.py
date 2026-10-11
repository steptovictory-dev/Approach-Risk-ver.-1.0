"""Locate repository configuration for source or editable package usage."""
from pathlib import Path
import os

def project_root():
    configured=os.environ.get('APPROACH_RISK_PROJECT_ROOT')
    if configured:
        root=Path(configured).resolve()
        if not (root/'configs/datasets.json').is_file():raise ValueError('Invalid APPROACH_RISK_PROJECT_ROOT')
        return root
    for location in [Path.cwd(),Path(__file__).resolve().parent]:
        for root in [location,*location.parents]:
            if (root/'configs/datasets.json').is_file() and (root/'pyproject.toml').is_file():return root
    raise FileNotFoundError('Set APPROACH_RISK_PROJECT_ROOT to the project containing configs/datasets.json')
