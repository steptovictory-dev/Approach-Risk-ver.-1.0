"""Versioned data references; local storage can change without changing experiments."""
from pathlib import Path
from dataclasses import dataclass
import hashlib,json
import pandas as pd
from approach_risk.paths import project_root

ROOT=project_root()
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

@dataclass(frozen=True)
class Dataset:
    name:str
    directory:Path
    manifest_path:Path
    feature_registry_path:Path
    protocol_path:Path
    manifest_sha256:str
    feature_registry_sha256:str
    def manifest(self):
        if sha(self.manifest_path)!=self.manifest_sha256:raise ValueError('Changed dataset manifest')
        if sha(self.feature_registry_path)!=self.feature_registry_sha256:raise ValueError('Changed feature registry')
        m=json.loads(self.manifest_path.read_text(encoding='utf8'))
        if sha(self.protocol_path)!=m['protocol_sha256']:raise ValueError('Changed preparation protocol')
        return m
    def iter_split(self,split,gap=False,columns=None):
        if split not in {'train','validation','test'}:raise ValueError('Unknown split')
        for entry in self.manifest()['files']:
            path=self.directory/entry['file']
            if sha(path)!=entry['sha256']:raise ValueError(f'Changed partition: {path.name}')
            required=['temporal_split','gap_status']
            fields=None if columns is None else list(dict.fromkeys(columns+required))
            f=pd.read_parquet(path,columns=fields)
            if len(f)!=entry['rows']:raise ValueError('Changed row count')
            keep=f.temporal_split.eq(split)
            if gap:keep &= f.gap_status.eq('retained')
            if keep.any():yield f.loc[keep,columns].copy() if columns is not None else f.loc[keep].copy()

def resolve(relative):
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):raise ValueError('Registry path outside project')
    return path

def dataset(name='europe_c40_model_v1'):
    config=json.loads((ROOT/'configs/datasets.json').read_text(encoding='utf8'))['datasets'][name]
    return Dataset(name=name,**{k:resolve(config[k]) for k in ['directory','manifest_path','feature_registry_path','protocol_path']},
                   **{k:config[k] for k in ['manifest_sha256','feature_registry_sha256']})
