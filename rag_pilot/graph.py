#!/usr/bin/env python3
"""Provenance-bearing graph from selected export occurrences."""
import argparse
import json
import sqlite3
from pathlib import Path

SCHEMA = '''
CREATE TABLE entity(id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL);
CREATE TABLE edge(id INTEGER PRIMARY KEY, subject TEXT NOT NULL REFERENCES entity(id), predicate TEXT NOT NULL, object TEXT NOT NULL REFERENCES entity(id), occurrence_id TEXT NOT NULL, evidence TEXT NOT NULL, method TEXT NOT NULL, status TEXT NOT NULL, extraction_version TEXT);
CREATE INDEX edge_subject ON edge(subject);
CREATE INDEX edge_object ON edge(object);
'''

def add_entity(db, id_, kind, label):
    db.execute('INSERT OR IGNORE INTO entity VALUES (?,?,?)', (id_,kind,label))

def add_edge(db, source, pred, target, occ, evidence, method, version):
    db.execute('INSERT INTO edge(subject,predicate,object,occurrence_id,evidence,method,status,extraction_version) VALUES (?,?,?,?,?,?,?,?)', (source,pred,target,occ,evidence,method,'unreviewed',version))

def build(selection, dbpath):
    path=Path(dbpath)
    if path.exists(): raise ValueError('graph database exists')
    db=sqlite3.connect(path)
    try:
        db.executescript(SCHEMA)
        with db, open(selection, encoding='utf-8') as f:
            for line in f:
                item=json.loads(line); r=item['record']; occ=item['occurrence_id']
                p='passage:'+occ; add_entity(db,p,'passage',r.get('locator') or occ)
                ev=json.dumps({'input_path':item['input_path'],'input_sha256':item['input_sha256'],'line':item['line'],'passage_id':r.get('passage_id')},ensure_ascii=False)
                version=r.get('adapter_version') or r.get('representation')
                # Supplied identifiers are namespace-specific and are not equated with canonical entities.
                for field,kind,pred in [('work_id','work','member_of_work'),('document_id','document','member_of_document')]:
                    value=r.get(field)
                    if isinstance(value,str) and value:
                        target=kind+':'+value; add_entity(db,target,kind,value)
                        add_edge(db,p,pred,target,occ,ev,'directly_supplied',version)
        db.close()
    except BaseException:
        db.close(); path.unlink(missing_ok=True); raise

def neighbors(db, entity, depth=1, limit=50):
    if depth < 1 or depth > 3 or limit < 1 or limit > 500: raise ValueError('depth must be 1..3 and limit 1..500')
    seen={entity}; frontier=[(entity,[])]; out=[]
    while frontier and len(out)<limit:
        node,path=frontier.pop(0)
        if len(path)>=depth: continue
        rows=db.execute('SELECT id,subject,predicate,object,occurrence_id,evidence,method,status,extraction_version FROM edge WHERE subject=? OR object=? ORDER BY id LIMIT ?', (node,node,limit)).fetchall()
        for row in rows:
            edge=dict(zip(('id','subject','predicate','object','occurrence_id','evidence','method','status','extraction_version'),row)); edge['evidence']=json.loads(edge['evidence'])
            other=edge['object'] if edge['subject']==node else edge['subject']
            if other in seen: continue
            seen.add(other); newpath=path+[edge]; out.append({'entity':other,'path':newpath})
            if len(out)>=limit: break
            frontier.append((other,newpath))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__); s=p.add_subparsers(dest='cmd',required=True)
    b=s.add_parser('build'); b.add_argument('--selection',required=True); b.add_argument('--db',required=True)
    q=s.add_parser('inspect'); q.add_argument('--db',required=True); q.add_argument('entity'); q.add_argument('--depth',type=int,default=1); q.add_argument('--limit',type=int,default=50)
    a=p.parse_args()
    if a.cmd=='build': build(a.selection,a.db)
    else:
        db=sqlite3.connect(f'file:{Path(a.db).resolve()}?mode=ro',uri=True)
        row=db.execute('SELECT id,kind,label FROM entity WHERE id=?',(a.entity,)).fetchone()
        if row is None: p.error('unknown entity')
        print(json.dumps({'entity':dict(zip(('id','kind','label'),row)), 'neighbors':neighbors(db,a.entity,a.depth,a.limit)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
