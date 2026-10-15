"""Enumerate implementation functions and verify all 100 output routes."""
import ast
import json
from pathlib import Path
import numpy as np
from PIL import Image
from trace_stroke import prepare, trace, segment_inside

root=Path(__file__).resolve().parents[1]
source=root/'scripts/trace_stroke.py'
functions=[n.name for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
tests=(root/'scripts/test_trace_stroke.py').read_text()
referenced=[name for name in functions if name+'(' in tests]
rows=[]
for file in sorted((root/'images').glob('*.png')):
    im=Image.open(file)
    result=trace(im)
    mask,_,_,_=prepare(im)
    path=np.array(result['path'])
    adjacent=bool(np.all(np.max(np.abs(np.diff(path,axis=0)),axis=1)<=1))
    on_ink=bool(np.all(mask[path[:,1],path[:,0]]))
    vector_on_ink=all(segment_inside(a,b,mask) for a,b in zip(result['vector'],result['vector'][1:]))
    repeats=[i for i,e in enumerate(result['edges']) if e['uses']==2]
    rows.append(dict(file=file.name,status=result['status'],chain_coverage=result['chain_coverage'],adjacent=adjacent,on_ink=on_ink,vector_on_ink=vector_on_ink,repeated_edges=repeats))
report=dict(functions=functions,functions_referenced_by_tests=referenced,
            reference_coverage_percent=100*len(referenced)/len(functions),
            implementation_source='scripts/trace_stroke.py',
            note='Function-reference coverage is not branch coverage. Generic behavior mapper extracted zero nodes; this AST inventory is the explicit supplement.',
            images=len(rows),complete=sum(r['status']=='complete' for r in rows),
            continuity_pass=sum(r['adjacent'] and r['on_ink'] and r['vector_on_ink'] for r in rows),rows=rows)
(root/'fresh_output/validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
print('Incomplete:',[(r['file'],r['chain_coverage']) for r in rows if r['status']!='complete'])
