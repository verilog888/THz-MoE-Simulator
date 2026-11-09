"""Stable, atomic portable artifacts."""
import csv
import gzip
import json
from pathlib import Path


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)


def write_gzip(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as handle:
        handle.write(json.dumps(value,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
    temporary.replace(path)


def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as handle:return json.load(handle)


def write_csv(path,rows):
    rows=list(rows)
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,separators=(',',':'),ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
