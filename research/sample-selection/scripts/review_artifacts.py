"""Инвентаризация исследования и проверка согласованности генераторов."""
from pathlib import Path
import ast,contextlib,hashlib,io,json,runpy
from unittest.mock import patch
import nbformat

root=Path(__file__).resolve().parents[1]
inventory=[]
for folder in ['','scripts','notebooks','reports','literature','figures','paper']:
    for path in sorted((root/folder).glob('*')):
        if path.is_file() and path.name!='research_inventory.json':
            data=path.read_bytes()
            inventory.append({'file':path.relative_to(root).as_posix(),'bytes':len(data),
                              'sha256':hashlib.sha256(data).hexdigest()})
sources=[]
for path in sorted((root/'scripts').glob('*.py')):
    ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
for path in sorted((root/'scripts').glob('build*notebook.py')):
    captured={}
    def capture(notebook,destination,*args,**kwargs):
        captured[Path(destination).name]=notebook
    with patch.object(nbformat,'write',side_effect=capture),contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(path))
    assert len(captured)==1,path
    for name,generated in captured.items():
        saved=nbformat.read(root/'notebooks'/name,as_version=4)
        matching=len(saved.cells)==len(generated.cells) and all(
            a.cell_type==b.cell_type and a.source==b.source for a,b in zip(saved.cells,generated.cells))
        errors=sum(o.output_type=='error' for c in saved.cells if c.cell_type=='code' for o in c.outputs)
        for cell in saved.cells:
            if cell.cell_type=='code':ast.parse(cell.source)
        sources.append({'generator':path.name,'notebook':name,'source_matches_generator':matching,
                        'cells':len(saved.cells),'error_outputs':errors,
                        'all_code_cells_executed':all(c.execution_count is not None for c in saved.cells if c.cell_type=='code')})
report={'scope':'public research files; local data and checkpoints are separate',
        'local_data_files':sum(p.is_file() for p in (root/'data').rglob('*')),
        'local_checkpoint_files':sum(p.is_file() for p in (root/'Checkpoint').glob('*')),
        'inventory':inventory,'notebook_sources':sources,'python_syntax_passed':True,
        'scientific_review':'reports/research_review.md'}
(root/'reports/research_inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for row in sources:print(row)
assert all(x['source_matches_generator'] and x['error_outputs']==0 and x['all_code_cells_executed'] for x in sources)
print(f'Inventory: {len(inventory)} public files')
