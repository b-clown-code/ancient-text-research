#!/usr/bin/env python3
"""Build document evidence and correspondence tables from the file inventory."""
import csv
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'catalog'
XML_LANG='{http://www.w3.org/XML/1998/namespace}lang'

def label(e): return ' '.join(' '.join(e.itertext()).split())
def local(e): return e.tag.rsplit('}',1)[-1]

def tt_metadata(line):
    # TreeTagger headers are SGML, including repeated attributes in some releases.
    # Preserve repeated values as evidence instead of treating SGML as strict XML.
    class Header(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag=='meta': self.attrs=attrs
    parser=Header(convert_charrefs=True); parser.attrs=[]; parser.feed(line)
    values={}; repeated={}
    for key,value in parser.attrs:
        if key in values: repeated.setdefault(key,[values[key]]).append(value)
        else: values[key]=value
    return values,repeated

def main():
    global ROOT, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT, help='workspace containing source directories')
    parser.add_argument('--out', type=Path, default=OUT, help='catalog output directory')
    args = parser.parse_args()

    # Only a full build needs the omitted workspace reconciliation helpers.
    try:
        try:
            from curation_schema import reconcile_existing_documents, table_exists
        except ModuleNotFoundError:
            from tools.curation_schema import reconcile_existing_documents, table_exists
    except ImportError as exc:
        raise SystemExit(
            "document catalog build requires the workspace curation_schema "
            "module, which is not part of this curated snapshot"
        ) from exc

    ROOT = args.root.resolve()
    OUT = args.out.resolve()
    db=sqlite3.connect(OUT/'corpus.sqlite')
    db.executescript('''
    DROP TABLE IF EXISTS cts_entries;
    DROP TABLE IF EXISTS documents;
    DROP TABLE IF EXISTS source_relationships;
    CREATE TABLE cts_entries(metadata_path TEXT,urn TEXT,work_urn TEXT,kind TEXT,language TEXT,title TEXT,description TEXT);
    CREATE TABLE documents(id INTEGER PRIMARY KEY,file_path TEXT,record_locator TEXT,title TEXT,author TEXT,language_hint TEXT,language_basis TEXT,work_urn TEXT,edition_urn TEXT,representation_family TEXT,license_evidence_json TEXT,date_evidence_json TEXT,composition_status TEXT,admission TEXT,evidence_basis TEXT);
    CREATE TABLE source_relationships(source_a TEXT,source_b TEXT,relationship TEXT,evidence TEXT);
    ''')
    relationships=[
      ('gretil-corpus','GRETIL-mirror','body-extraction derivative','gretil-corpus/README.md and catalog.csv pin mirror commit'),
      ('CTSAncientGreekXML','canonical-greekLit','tokenized derivative overlap','CTSAncientGreekXML/README.md'),
      ('CTSAncientGreekXML','First1KGreek','tokenized derivative overlap','CTSAncientGreekXML/README.md'),
      ('oracc-parser','oracc-data','parser and acquired raw data','oracc-parser/README.md; oracc-data/ZENODO_RECORD.json'),
      ('latin-literature-170M','openmgh','possible upstream collection overlap; not verified duplicates','Corpus Corporum source lineage; review extracted text before grouping'),
    ]
    db.executemany('INSERT INTO source_relationships VALUES (?,?,?,?)',relationships)
    cts={}
    for (path,) in db.execute("SELECT path FROM files WHERE role='cts_metadata'").fetchall():
        try: root=ET.parse(ROOT/path).getroot()
        except ET.ParseError: continue
        for e in root.iter():
            if local(e) not in ('edition','translation'): continue
            urn=e.get('urn'); work=e.get('workUrn') or root.get('urn')
            if not urn: continue
            title=next((label(x) for x in e if local(x)=='label'),None)
            desc=next((label(x) for x in e if local(x)=='description'),None)
            lang=e.get(XML_LANG)
            # CTS versions frequently encode language in their conventional suffix;
            # keep that inference distinct from an explicit XML language attribute.
            if not lang:
                m=re.search(r'-([a-z]{3})\d*$',urn)
                if m: lang=m.group(1)
            cts[urn]=(work,title,lang,local(e))
            db.execute('INSERT INTO cts_entries VALUES (?,?,?,?,?,?,?)',(path,urn,work,local(e),lang,title,desc))
    gretil={}
    with (ROOT/'gretil-corpus/catalog.csv').open(encoding='utf-8',newline='') as f:
        gretil={r['path']:r for r in csv.DictReader(f)}
    candidates=db.execute("SELECT source,path,role,representation_family,metadata_json FROM files WHERE role IN ('primary_text_candidate','edition_or_translation_candidate','edition_candidate','derived_text')").fetchall()
    def add(path,locator,title,author,language,basis,work,edition,family,licenses,dates,status,evidence):
        db.execute('INSERT INTO documents(file_path,record_locator,title,author,language_hint,language_basis,work_urn,edition_urn,representation_family,license_evidence_json,date_evidence_json,composition_status,admission,evidence_basis) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
          (path,locator,title,author,language,basis,work,edition,family,json.dumps(licenses,ensure_ascii=False),json.dumps(dates,ensure_ascii=False),status,'pending_review',evidence))
    for source,path,role,family,raw in candidates:
        m=json.loads(raw); p=Path(path); rel=p.relative_to(source).as_posix()
        titles=m.get('titles',[]); authors=m.get('authors',[])
        title=titles[0]['text'] if titles else None
        author='; '.join(a['text'] for a in authors) or None
        langs=[]
        for item in m.get('languages',[]):
            a=item['attributes']; value=a.get('ident') or a.get('mainLang')
            if value and value not in langs: langs.append(value)
        language=';'.join(langs) or None; basis='TEI header language list; may include metadata and translation languages' if langs else None
        urn=next((u for u in cts if u.rsplit(':',1)[-1]==p.stem),None) if role=='edition_or_translation_candidate' else None
        work=None
        if urn:
            work,ctitle,clang,kind=cts[urn]
            title=title or ctitle
            if clang: language=clang; basis='CTS metadata or conventional version suffix'
        license=m.get('licenses',[]); dates=m.get('dates',[]); status='unverified'
        evidence='file inventory; source TEI header where available'
        if source=='gretil-corpus' and rel in gretil:
            g=gretil[rel]; title=g['title']; author=g['author'] or None
            license=[{'claim':g['license'],'evidence':'gretil-corpus/catalog.csv'}]
            language=p.name.split('_',1)[0]; basis='source filename prefix'
            evidence='gretil-corpus/catalog.csv; bibliographic metadata explicitly unverified upstream'
            if rel.startswith('7_later/'): status='flagged_later_by_source_directory'
        if source=='suttacentral-bilara': language=p.parts[2]; basis='source directory code'
        if source=='ancient-greek-texts': language='grc'; basis='grc source directory'
        if source=='coptic-scriptorium' and p.suffix=='.tt':
            with (ROOT/p).open(encoding='utf-8') as f: line=f.readline().strip()
            try:
                a,repeated=tt_metadata(line)
                if not a: raise ET.ParseError('Missing meta tag')
                title=a.get('title') or a.get('msitem_title'); author=a.get('author')
                language=a.get('language'); basis='TreeTagger source document language attribute'
                license=[{'claim':a.get('license'),'source_info':a.get('source_info')}]
                dates=[{'context':'TreeTagger manuscript origDate','attributes':{k:v for k,v in a.items() if k.startswith('origdate')}},
                       {'context':'digital edition version','value':a.get('version_date')}]
                doc_urn=a.get('document_cts_urn')
                if doc_urn:
                    urn=':'.join(doc_urn.split(':')[:4]); work=urn.rsplit('.',1)[0]
                evidence=json.dumps({'basis':'TreeTagger SGML document metadata','document_cts_urn':doc_urn,'redundant':a.get('redundant'),'witness_note':a.get('witness'),'repeated_attributes':repeated},ensure_ascii=False)
            except ET.ParseError:
                evidence='TreeTagger metadata parse failed; review first line'
        add(path,None,title,author,language,basis,work,urn,family,license,dates,status,evidence)
    # Catalog each raw Latin CSV row without duplicating its long text field.
    csv.field_size_limit(sys.maxsize)
    with (ROOT/'latin-literature-170M/latin_raw.csv').open(encoding='utf-8-sig',newline='') as f:
        for n,row in enumerate(csv.DictReader(f),1):
            add('latin-literature-170M/latin_raw.csv',f'row:{n};source-id:{row.get("", "")}',
                row.get('title'),row.get('author'),'lat','source collection claim; mixed-language passages possible',None,None,None,[],
                [{'raw_year':row.get('year'),'date_type':'unresolved_source_year'}],'unverified','source CSV columns; year meaning unverified')
    if table_exists(db, 'document_identities') and table_exists(db, 'raw_inputs'):
        db.execute('UPDATE document_identities SET active=0')
        reconcile_existing_documents(db, deactivate_missing=False)
    db.executescript('CREATE INDEX documents_path ON documents(file_path); CREATE INDEX documents_cts ON documents(work_urn);')
    db.commit()
    with (OUT/'documents.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); cur=db.execute('SELECT file_path,record_locator,title,author,language_hint,language_basis,work_urn,edition_urn,composition_status,admission FROM documents ORDER BY id')
        w.writerow([d[0] for d in cur.description]); w.writerows(cur)
    with (OUT/'languages.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['language_hint','basis','document_records'])
        w.writerows(db.execute("SELECT coalesce(language_hint,'unknown'),coalesce(language_basis,'unknown'),count(*) FROM documents GROUP BY language_hint,language_basis ORDER BY count(*) DESC"))
    with (OUT/'review_queue.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['path','issue','evidence'])
        w.writerows(db.execute("SELECT path,'tei_header_parse_error',json_extract(metadata_json,'$.error') FROM files WHERE json_extract(metadata_json,'$.header_status')='parse_error' ORDER BY path"))
        w.writerows(db.execute("SELECT file_path,composition_status,evidence_basis FROM documents WHERE composition_status != 'unverified' ORDER BY file_path"))
    with (OUT/'source_relationships.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['source_a','source_b','relationship','evidence'])
        w.writerows(db.execute('SELECT * FROM source_relationships ORDER BY source_a,source_b'))
    counts={k:db.execute(f'SELECT count(*) FROM {k}').fetchone()[0] for k in ('cts_entries','documents','source_relationships')}
    report=OUT/'REPORT.md'
    content=report.read_text().split('\n## Document evidence catalog\n')[0]
    content+='\n## Document evidence catalog\n\n'
    content+=f"Document/representation records: {counts['documents']:,}. CTS edition/translation entries: {counts['cts_entries']:,}. Source relationships recorded: {counts['source_relationships']}. These are candidate records, not unique works.\n\n"
    content+='See documents.csv for titles, authors, language hints, CTS work/edition IDs, and unresolved composition status. The database preserves date contexts and license evidence without promoting them to verified composition dates or admission. languages.csv reports source hints, not language detection. Latin CSV year values retain an unresolved date type.\n'
    content+='\nTEI header language lists may include modern metadata or translation languages; they are not counts of primary-language texts. Coptic TreeTagger manuscript dates remain witness dates.\n'
    failures=db.execute("SELECT count(*) FROM files WHERE json_extract(metadata_json,'$.header_status')='parse_error'").fetchone()[0]
    content+=f'\nTEI header parse failures flagged: {failures:,}. Missing, non-TEI, or oversized headers remain unresolved. No failed parse is silently admitted.\n'
    report.write_text(content)
    print(json.dumps(counts)); db.close()

if __name__=='__main__': main()
