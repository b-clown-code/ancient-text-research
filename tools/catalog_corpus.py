#!/usr/bin/env python3
"""Offline, non-destructive corpus inventory. No third-party dependencies."""
import collections
import csv
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'catalog'
# These are provisional source claims. Per-document evidence takes precedence.
SOURCES = {
 'ancient-greek-texts': ('PDM text; CC0 scaffolding', 'greek-open-editions', 'candidate'),
 'canonical-greekLit': ('CC BY-SA 4.0', 'perseus-greek', 'candidate'),
 'canonical-latinLit': ('CC BY-SA 4.0', 'perseus-latin', 'candidate'),
 'First1KGreek': ('CC BY-SA 4.0', 'first1k-greek', 'candidate'),
 'CTSAncientGreekXML': ('CC BY-NC 4.0', 'perseus-first1k-derivative', 'restricted'),
 'GRETIL-mirror': ('Per-file; predominantly CC BY-NC-SA 4.0', 'gretil', 'mixed'),
 'gretil-corpus': ('783 NC-SA; one BY-SA per catalog.csv', 'gretil', 'mixed'),
 'aed-tei': ('CC BY-SA 4.0', 'aed-tla', 'candidate'),
 'Semantic-Role-Labeler': ('MIT software; embedded upstream data unresolved', 'etcsl-cdli-derivative', 'unresolved'),
 'oracc-parser': ('MIT software; outputs are derivatives', 'oracc', 'software'),
 'latin-literature-170M': ('Corpus Corporum mixed upstream rights', 'corpus-corporum', 'unresolved'),
 'digital-syriac-corpus': ('CC BY 4.0', 'digital-syriac', 'candidate'),
 'coptic-scriptorium': ('CC BY with per-corpus exceptions', 'coptic-scriptorium', 'mixed'),
 'papyri-info': ('CC BY 3.0', 'idp', 'candidate'),
 'openmgh': ('CC BY 4.0', 'openmgh', 'candidate'),
 'menota': ('Corpus CC BY-SA 4.0; verify individual files', 'menota', 'mixed'),
 'tanzil-quran': ('CC BY 3.0 plus no-modification terms', 'tanzil', 'special_terms'),
 'oracc-data': ('Per-project licenses; release checksum verified', 'oracc', 'mixed'),
 'suttacentral-bilara': ('Translations CC0; root editions require review', 'suttacentral', 'mixed'),
}

def classify(source, rel):
    p = Path(rel); s = rel.lower(); ext = p.suffix.lower()
    if ext == '.zip': return 'archive'
    if p.name == '__cts__.xml': return 'cts_metadata'
    if source == 'papyri-info':
        first = p.parts[0]
        if first in ('DDbDP', 'DCLP', 'APD') and ext == '.xml': return 'primary_text_candidate'
        if first == 'Translations': return 'translation_reference'
        if first == 'Historical': return 'historical_snapshot'
        return 'metadata_or_support'
    if source == 'suttacentral-bilara':
        if s.startswith('translation/'): return 'translation_reference'
        if s.startswith('root/') and ext == '.json':
            return 'modern_editorial' if s.startswith('root/en/') else 'primary_text_candidate'
        return 'metadata_or_support'
    if source == 'ancient-greek-texts':
        return 'primary_text_candidate' if s.startswith('grc/') and ext == '.txt' else 'alternate_or_support'
    if source == 'aed-tei' and s.startswith('files/') and ext == '.xml':
        if p.stem.endswith(('_st', '_wt')): return 'translation_or_gloss'
        if p.stem.endswith('_hiero'): return 'alternate_representation'
        return 'primary_text_candidate'
    if source == 'coptic-scriptorium':
        if ext == '.tt': return 'primary_text_candidate'
        if ext in ('.xml', '.conllu', '.annis'): return 'alternate_representation'
    if source == 'digital-syriac-corpus':
        return 'primary_text_candidate' if s.startswith('data/tei/') and ext == '.xml' else 'metadata_or_support'
    if source in ('canonical-greekLit','canonical-latinLit','First1KGreek'):
        return 'edition_or_translation_candidate' if s.startswith('data/') and ext == '.xml' else 'metadata_or_support'
    if source == 'GRETIL-mirror' and ext in ('.xml','.htm','.html','.txt'): return 'edition_candidate'
    if source == 'gretil-corpus' and ext == '.txt': return 'edition_candidate'
    if source in ('menota','openmgh') and s.startswith('extracted/') and ext == '.xml': return 'primary_text_candidate'
    if source == 'tanzil-quran' and p.name.startswith('quran-') and ext == '.xml': return 'primary_text_candidate'
    if source == 'latin-literature-170M' and ext == '.csv': return 'bulk_text_container'
    if source == 'CTSAncientGreekXML' and s.startswith('texts/') and ext == '.xml': return 'derived_text'
    if source in ('Semantic-Role-Labeler','oracc-parser'): return 'derivative_or_software'
    return 'metadata_or_support'

def representation(source, rel):
    p = Path(rel)
    if source == 'aed-tei' and rel.startswith('files/'):
        return 'aed:' + re.sub(r'_(st|wt|hiero)$', '', p.stem)
    if source == 'coptic-scriptorium': return 'coptic:' + p.stem
    if source == 'tanzil-quran' and p.name.startswith('quran-'): return 'tanzil:quran'
    if source == 'ancient-greek-texts' and rel.startswith('grc/'):
        return 'greek-open:' + re.sub(r'\.grc$', '', p.stem)
    return None

def header_metadata(prefix):
    """Parse only a closed TEI header, without resolving external entities."""
    s = prefix.decode('utf-8-sig', errors='replace')
    match = re.search(r'<(?:\w+:)?teiHeader\b.*?</(?:\w+:)?teiHeader\s*>', s, re.S)
    if not match: return {'header_status': 'not_found_within_256KiB'}
    fragment = match.group()
    if '<!ENTITY' in fragment: return {'header_status': 'entity_declaration_rejected'}
    # Headers may inherit namespace prefixes from TEI root.
    ns = dict(re.findall(r'xmlns:([\w.-]+)=[\"\x27]([^\"\x27]+)', s[:match.start()]))
    wrapper = '<wrapper xmlns="http://www.tei-c.org/ns/1.0" ' + ' '.join(
        f'xmlns:{k}="{v}"' for k,v in ns.items()) + '>' + fragment + '</wrapper>'
    try: root = ET.fromstring(wrapper)
    except ET.ParseError as e: return {'header_status': 'parse_error', 'error': str(e)}
    def local(e): return e.tag.rsplit('}',1)[-1]
    def entries(names):
        return [{'text': ' '.join(' '.join(e.itertext()).split()), 'attributes': dict(e.attrib)}
                for e in root.iter() if local(e) in names]
    # Preserve contexts for dates: no inferred composition date.
    dates = []
    def walk(e, path):
        path = path + '/' + local(e)
        if local(e) in ('date','origDate'): dates.append({'context':path,'text':' '.join(' '.join(e.itertext()).split()),'attributes':dict(e.attrib)})
        for c in e: walk(c,path)
    walk(root,'')
    return {'header_status':'parsed', 'titles':entries({'title'}), 'authors':entries({'author'}),
            'languages':entries({'language','textLang'}), 'licenses':entries({'licence','license'}),
            'identifiers':entries({'idno'}), 'dates': dates,
            'responsibilities':entries({'respStmt'}), 'evidence_status':'unverified_source_metadata'}

def create_review_ledger(db):
    """Create v1 review tables and carry append-only state across rebuilds."""
    create_v1_tables(db)
    ensure_parallel_schema(db)
    old = OUT / 'corpus.sqlite'
    main_path = Path(db.execute('PRAGMA database_list').fetchone()[2]).resolve()
    if old.exists() and old.resolve() != main_path:
        prior = sqlite3.connect(old)
        try:
            copy_curation_state(prior, db)
            copy_parallel_state(prior, db)
        finally:
            prior.close()

def main():
    global ROOT, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT, help='workspace containing source directories')
    parser.add_argument('--out', type=Path, default=OUT, help='catalog output directory')
    args = parser.parse_args()

    # These workspace-only integrations are needed for a full inventory build,
    # but should not prevent standalone --help or imports of pure helpers.
    try:
        try:
            from curation_schema import add_inventory_inputs, copy_curation_state, create_v1_tables, reconcile_scanned_archive_members
            from curation_runtime import require_free_space
            from parallel_schema import ensure_parallel_schema, copy_parallel_state
        except ModuleNotFoundError:
            from tools.curation_schema import add_inventory_inputs, copy_curation_state, create_v1_tables, reconcile_scanned_archive_members
            from tools.curation_runtime import require_free_space
            from tools.parallel_schema import ensure_parallel_schema, copy_parallel_state
    except ImportError as exc:
        raise SystemExit(
            "full catalog build requires workspace curation runtime modules "
            "(curation_schema, curation_runtime, and parallel_schema), which "
            "are not part of this curated snapshot"
        ) from exc

    ROOT = args.root.resolve()
    OUT = args.out.resolve()
    OUT.mkdir(exist_ok=True)
    require_free_space(ROOT, 10.0)
    tmp = OUT / 'corpus.build.sqlite'
    if tmp.exists(): raise SystemExit('Build database already exists; review previous interrupted run.')
    db=sqlite3.connect(tmp)
    db.executescript('''PRAGMA journal_mode=OFF;
    CREATE TABLE sources(source TEXT PRIMARY KEY,origin TEXT,commit_sha TEXT,license_claim TEXT,provenance_family TEXT,rights_review TEXT,admission TEXT);
    CREATE TABLE files(id INTEGER PRIMARY KEY,source TEXT,path TEXT UNIQUE,bytes INTEGER,mtime_ns INTEGER,format TEXT,role TEXT,sha256 TEXT,representation_family TEXT,admission TEXT,metadata_json TEXT);
    CREATE TABLE inventory_archive_members(archive_path TEXT,member_path TEXT,member_chain_json TEXT NOT NULL,raw_sha256 TEXT,bytes INTEGER,compressed_bytes INTEGER,crc32 TEXT,is_directory INTEGER);
    CREATE TABLE errors(path TEXT,stage TEXT,error TEXT);
    CREATE TABLE run_info(key TEXT PRIMARY KEY,value TEXT);
    ''')
    create_review_ledger(db)
    db.execute('INSERT INTO run_info VALUES (?,?)',('started_utc',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    registry=[]; total=0
    for source,(license,family,rights) in SOURCES.items():
        base=ROOT/source
        if not base.is_dir(): continue
        def git(*args):
            r=subprocess.run(['git','-C',str(base),*args],capture_output=True,text=True)
            return r.stdout.strip() if r.returncode==0 else None
        origin=git('remote','get-url','origin'); commit=git('rev-parse','HEAD')
        record=dict(source=source,origin=origin,commit_sha=commit,license_claim=license,provenance_family=family,rights_review=rights,admission='pending_review')
        registry.append(record)
        db.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?)',tuple(record.values()))
        n=0
        for folder, dirs, names in os.walk(base):
            dirs[:]=sorted(d for d in dirs if d not in ('.git','__pycache__','.venv','node_modules') and not Path(folder,d).is_symlink())
            for name in sorted(names):
                p=Path(folder,name)
                if p.is_symlink(): continue
                rel=p.relative_to(base).as_posix(); path=p.relative_to(ROOT).as_posix()
                try:
                    st=p.stat(); role=classify(source,rel); h=hashlib.sha256(); prefix=b''
                    with p.open('rb') as f:
                        while chunk:=f.read(1024*1024):
                            h.update(chunk)
                            if len(prefix)<262144: prefix+=chunk[:262144-len(prefix)]
                    metadata={}
                    if p.suffix.lower()=='.xml' and role in ('primary_text_candidate','edition_or_translation_candidate','edition_candidate','derived_text'):
                        metadata=header_metadata(prefix)
                    if role=='cts_metadata': metadata={'cts_urn_candidates':sorted(set(re.findall(r'urn:cts:[^\s\"<>]+',prefix.decode('utf-8',errors='replace')))), 'evidence_status':'correspondence_only'}
                    db.execute('INSERT INTO files(source,path,bytes,mtime_ns,format,role,sha256,representation_family,admission,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?)',
                               (source,path,st.st_size,st.st_mtime_ns,p.suffix.lower() or '[none]',role,h.hexdigest(),representation(source,rel),'pending_review',json.dumps(metadata,ensure_ascii=False)))
                    if p.suffix.lower()=='.zip':
                        try:
                            def scan_zip(handle, chain_prefix, depth=0):
                                # Occurrences are local to this exact archive,
                                # including when its parent name is repeated.
                                occurrence_by_name = collections.defaultdict(int)
                                with zipfile.ZipFile(handle) as z:
                                    for info in z.infolist():
                                        occurrence = occurrence_by_name[info.filename]
                                        occurrence_by_name[info.filename] += 1
                                        chain = chain_prefix + [{'member_path': info.filename, 'occurrence_index': occurrence}]
                                        yield (path, info.filename, json.dumps(chain, ensure_ascii=False, sort_keys=True, separators=(',', ':')), None, info.file_size, info.compress_size, f'{info.CRC:08x}', int(info.is_dir()))
                                        if info.is_dir() or depth >= 3 or not info.filename.lower().endswith(('.zip', '.jar', '.tar')):
                                            continue
                                        if info.file_size > 64 * 1024 * 1024:
                                            continue
                                        require_free_space(ROOT, 10.0, 64 * 1024 * 1024)
                                        with tempfile.NamedTemporaryFile(dir=OUT, prefix='.nested-', suffix='.zip') as nested:
                                            with z.open(info) as source_member:
                                                shutil.copyfileobj(source_member, nested, length=1024 * 1024)
                                            nested.flush()
                                            nested.seek(0)
                                            try:
                                                yield from scan_zip(nested, chain, depth + 1)
                                            except zipfile.BadZipFile:
                                                pass

                            db.executemany('INSERT INTO inventory_archive_members(archive_path,member_path,member_chain_json,raw_sha256,bytes,compressed_bytes,crc32,is_directory) VALUES (?,?,?,?,?,?,?,?)', scan_zip(p, []))
                        except (zipfile.BadZipFile,OSError) as e: db.execute('INSERT INTO errors VALUES (?,?,?)',(path,'zip_directory',str(e)))
                    n+=1; total+=1
                    if total%5000==0:
                        db.commit()
                        require_free_space(ROOT, 10.0)
                except OSError as e: db.execute('INSERT INTO errors VALUES (?,?,?)',(path,'read',str(e)))
        db.commit(); print(f'{source}: {n:,} files indexed',flush=True)
    db.execute('UPDATE raw_inputs SET active=0')
    reserve_check = lambda: require_free_space(ROOT, 10.0)
    reconciled_inputs = add_inventory_inputs(db, reserve_check=reserve_check)
    reconciled_members = reconcile_scanned_archive_members(db, reserve_check=reserve_check)
    db.commit()
    db.executescript('CREATE INDEX files_hash ON files(sha256); CREATE INDEX files_source_role ON files(source,role); CREATE INDEX files_representation ON files(representation_family);')
    rows=db.execute('SELECT source,count(*),sum(bytes),sum(CASE WHEN role IN (\'primary_text_candidate\',\'edition_or_translation_candidate\',\'edition_candidate\') THEN 1 ELSE 0 END) FROM files GROUP BY source ORDER BY source').fetchall()
    with (OUT/'sources.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['source','payload_files','payload_bytes_excluding_git','candidate_files_not_works']); w.writerows(rows)
    with (OUT/'exact_duplicates.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['sha256','copies','bytes_each','redundant_bytes','example_paths'])
        for digest,copies,size in db.execute('SELECT sha256,count(*),max(bytes) FROM files WHERE bytes>0 GROUP BY sha256 HAVING count(*)>1 ORDER BY (count(*)-1)*max(bytes) DESC'):
            paths=[x[0] for x in db.execute('SELECT path FROM files WHERE sha256=? ORDER BY path LIMIT 8',(digest,))]
            w.writerow([digest,copies,size,(copies-1)*size,json.dumps(paths,ensure_ascii=False)])
    dup=db.execute('SELECT count(*),coalesce(sum((copies-1)*bytes),0) FROM (SELECT count(*) copies,max(bytes) bytes FROM files WHERE bytes>0 GROUP BY sha256 HAVING count(*)>1)').fetchone()
    errors=db.execute('SELECT count(*) FROM errors').fetchone()[0]
    archive_count=db.execute('SELECT count(*) FROM archive_members WHERE is_directory=0').fetchone()[0]
    report=['# Corpus inventory','',f'Indexed {total:,} files across {len(rows)} sources. Reconciled {reconciled_inputs:,} stable raw-input identities and {reconciled_members:,} archive-member identities. Payload bytes exclude Git objects. Counts are files, not distinct works or training examples. All admission remains pending review.','', '| Source | Files | Payload GiB | Candidate files |','|---|---:|---:|---:|']
    report += [f'| {s} | {n:,} | {b/2**30:.3f} | {c:,} |' for s,n,b,c in rows]
    report += ['',f'Exact nonempty duplicate groups: {dup[0]:,}. Repeated byte content: {dup[1]/2**20:.1f} MiB. This is not a deletion estimate: provenance and independent originals remain preserved.',f'ZIP member records (files): {archive_count:,}; nested archives are not expanded.',f'Read/archive errors: {errors}. See SQLite errors table.', '', '## Routing counts','', '| Role | Files |','|---|---:|']
    report += [f'| {r} | {n:,} |' for r,n in db.execute('SELECT role,count(*) FROM files GROUP BY role ORDER BY count(*) DESC')]
    report += ['', '## Next review steps','', '1. Reconcile existing source catalogs and CTS metadata into reviewed work/edition records.', '2. Resolve file-level rights exceptions and date evidence; keep witness and edition dates separate from composition.', '3. Extract primary text with source-specific adapters, excluding modern translation, apparatus, summaries, code, and repeated formats.', '4. Deduplicate extracted text and review edition/translation relationships.', '5. Establish whole-work leakage groups and holdouts before tokenizer or training exports.', '', 'No raw files were moved, changed, or deleted.']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    (OUT/'source_registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
    db.execute('INSERT INTO run_info VALUES (?,?)',('completed_utc',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    db.commit(); db.close()
    os.replace(tmp,OUT/'corpus.sqlite')
    print(f'Complete: {total:,} files; {errors} errors. {OUT / "REPORT.md"}',flush=True)

if __name__=='__main__': main()
