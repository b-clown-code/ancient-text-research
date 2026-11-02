#!/usr/bin/env python3
"""Source-preserving window index and evidence search for a frozen selection."""
import argparse, hashlib, json, sqlite3, sys, time
from pathlib import Path
import numpy as np
from dense import Encoder
from graph import neighbors

SCHEMA='''CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE passage(id TEXT PRIMARY KEY, record TEXT NOT NULL, input_path TEXT NOT NULL, input_sha256 TEXT NOT NULL, line INTEGER NOT NULL);
CREATE TABLE window(id INTEGER PRIMARY KEY, text TEXT NOT NULL, source TEXT, language TEXT, work_id TEXT, document_id TEXT, vector BLOB NOT NULL);
CREATE TABLE mapping(window_id INTEGER, passage_id TEXT, start INTEGER, end INTEGER, position INTEGER, PRIMARY KEY(window_id,passage_id,start));
CREATE INDEX wm ON mapping(window_id); CREATE INDEX mp ON mapping(passage_id); CREATE VIRTUAL TABLE lexical USING fts5(text, tokenize='unicode61 remove_diacritics 0');'''

def build(a):
    dbpath=Path(a.db)
    if dbpath.exists(): raise ValueError('index already exists')
    encoder=Encoder(a.model,a.revision,a.cache,batch_size=a.batch,
                    pooling=a.pooling, dimension=a.dimension,
                    trust_remote_code=a.trust_remote_code)
    if a.tokens<4 or a.tokens>encoder.max_length: raise ValueError('token limit exceeds model limit')
    db=sqlite3.connect(dbpath)
    db.execute('PRAGMA synchronous = NORMAL')
    db.execute('PRAGMA journal_mode = WAL')
    started=time.monotonic(); pending=[]; pending_passages=[]; current=None
    def flush_windows():
        nonlocal pending, pending_passages
        if not pending and not pending_passages: return
        vectors = encoder.encode_passages([x[0] for x in pending]) if pending else []
        with db:
            if pending_passages:
                db.executemany('INSERT INTO passage VALUES(?,?,?,?,?)', pending_passages)
                pending_passages = []
            for (text,meta,maps),vec in zip(pending,vectors):
                c=db.execute('INSERT INTO window(text,source,language,work_id,document_id,vector) VALUES(?,?,?,?,?,?)',(text,*meta,vec.astype('<f4').tobytes())); wid=c.lastrowid
                db.execute('INSERT INTO lexical(rowid,text) VALUES(?,?)',(wid,text))
                db.executemany('INSERT INTO mapping VALUES(?,?,?,?,?)',[(wid,*m,i) for i,m in enumerate(maps)])
        pending=[]
    def flush_current():
        nonlocal current
        if current:
            pending.append((current['text'],current['meta'],current['maps'])); current=None
            if len(pending)>=a.batch: flush_windows()
    try:
        db.executescript(SCHEMA)
        db.executemany('INSERT INTO metadata VALUES (?,?)', [
            ('model', str(encoder.model_path)),
            ('revision', encoder.revision),
            ('encoder_identity', encoder.get_index_identity()),
            ('embedding_dimension', str(encoder.dimension)),
            ('selection_sha256', hashlib.sha256(Path(a.selection).read_bytes()).hexdigest()),
        ])
        line_count=0
        with open(a.selection,encoding='utf8') as f:
            for line in f:
                line_count+=1
                x=json.loads(line); r=x['record']; pid=x['occurrence_id']; text=r['text']; meta=(r.get('source'),r.get('language'),r.get('work_id'),r.get('document_id'))
                pending_passages.append((pid,json.dumps(r,ensure_ascii=False),x['input_path'],x['input_sha256'],x['line']))
                if len(pending_passages) >= 2000:
                    with db:
                        db.executemany('INSERT INTO passage VALUES(?,?,?,?,?)', pending_passages)
                        pending_passages = []
                order=r.get('ordinal'); key=(x['input_sha256'],*meta)
                if encoder.tokens(text)>a.tokens:
                    flush_current()
                    tok=encoder.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
                    offsets=tok['offset_mapping']
                    max_tok=a.tokens-4
                    cur_idx=0
                    prev_end=0
                    while cur_idx<len(offsets):
                        cand_idx=min(cur_idx+max_tok,len(offsets))-1
                        while cand_idx>=cur_idx:
                            s_char=offsets[cur_idx][0] if cur_idx==0 else prev_end
                            e_char=offsets[cand_idx+1][0] if cand_idx+1<len(offsets) else len(text)
                            sub=text[s_char:e_char]
                            if encoder.tokens(sub)<=a.tokens:
                                break
                            cand_idx-=1
                        if cand_idx<cur_idx:
                            raise ValueError('cannot fit even one token')
                        pending.append((sub,meta,[(pid,s_char,e_char)]))
                        prev_end=e_char
                        cur_idx=cand_idx+1
                        if len(pending)>=a.batch:flush_windows()
                    continue
                join=current and current['key']==key and isinstance(order,int) and current['ordinal'] is not None and order==current['ordinal']+1 and encoder.tokens(current['text']+'\n'+text)<=a.tokens
                if not join: flush_current(); current={'text':text,'meta':meta,'key':key,'ordinal':order if isinstance(order,int) else None,'maps':[(pid,0,len(text))]}
                else:
                    current['text']+='\n'+text; current['maps'].append((pid,0,len(text))); current['ordinal']=order
                if line_count % 5000 == 0:
                    sys.stderr.write(f'Passages {line_count} (elapsed: {round(time.monotonic()-started,1)}s)...\n')
        flush_current()
        flush_windows()
        if pending_passages:
            with db:
                db.executemany('INSERT INTO passage VALUES(?,?,?,?,?)', pending_passages)
                pending_passages = []
        db.commit()
        print(json.dumps({'seconds':round(time.monotonic()-started,2),'passages':db.execute('select count(*) from passage').fetchone()[0],'windows':db.execute('select count(*) from window').fetchone()[0],'bytes':dbpath.stat().st_size,'dimension':encoder.dimension}))
    except BaseException:
        db.close(); encoder.close(); dbpath.unlink(missing_ok=True); raise
    db.close(); encoder.close()

class VectorMatrixCache:
    """Explicitly scoped dense-vector cache; call clear when its owner is done."""

    def __init__(self):
        self.connection = None
        self.version = None
        self.matrices = {}

    def clear(self):
        self.connection = None
        self.version = None
        self.matrices.clear()

    def get(self, db, clause='', params=()):
        # Never retain transaction-local vectors: rollback does not advance
        # total_changes or data_version. Also discard pre-transaction entries.
        if db.in_transaction:
            self.clear()
            return self._load(db, clause, params)
        data_version = db.execute('PRAGMA data_version').fetchone()[0]
        version = (data_version, db.total_changes)
        if self.connection is not db or self.version != version:
            self.clear()
            self.connection = db
            self.version = version
        key = (clause, tuple(params))
        if key in self.matrices:
            return self.matrices[key]
        result = self._load(db, clause, params)
        self.matrices[key] = result
        return result

    @staticmethod
    def _load(db, clause='', params=()):
        rows = db.execute('SELECT w.id, w.vector FROM window w WHERE 1=1' + clause, params).fetchall()
        if rows and hasattr(rows[0], 'keys'):
            ids = np.array([r['id'] for r in rows], dtype=np.int64)
            mat = np.stack([np.frombuffer(r['vector'], dtype='<f4') for r in rows])
        elif rows:
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            mat = np.stack([np.frombuffer(r[1], dtype='<f4') for r in rows])
        else:
            ids = np.empty(0, dtype=np.int64)
            metadata = dict(db.execute('SELECT key,value FROM metadata').fetchall())
            dimension = int(metadata.get('embedding_dimension', 384))
            mat = np.empty((0, dimension), dtype=np.float32)
        return ids, mat


def get_vector_matrix(db, clause='', params=(), cache=None):
    """Load vectors, optionally using a caller-owned connection-scoped cache."""
    if cache is not None:
        return cache.get(db, clause, params)
    return VectorMatrixCache._load(db, clause, params)


def validate_encoder_compatibility(db, encoder):
    """Reject dense queries whose encoder differs from the indexed passage encoder."""
    metadata = dict(db.execute('SELECT key,value FROM metadata').fetchall())
    expected = metadata.get('encoder_identity')
    if not expected:
        raise ValueError('index has no encoder identity; rebuild it before dense retrieval')
    actual = encoder.get_index_identity()
    if actual != expected:
        raise ValueError('query encoder is incompatible with the index passage encoder')
    expected_dimension = metadata.get('embedding_dimension')
    if expected_dimension is not None and int(expected_dimension) != int(encoder.dimension):
        raise ValueError('query encoder dimension is incompatible with the index')

def search_unified(
    db,
    query,
    mode='hybrid',
    model=None,
    revision=None,
    cache=None,
    encoder=None,
    graph_db=None,
    graph_seeds=3,
    graph_limit=40,
    graph_weight=0.1,
    min_dense_sim=None,
    limit=10,
    candidates=50,
    rrf=60,
    phrase=False,
    source=None,
    language=None,
    work_id=None,
    document_id=None,
    batch=64,
    reranker=None,
    rerank_candidates=50,
    task='modern_query_to_primary_evidence',
    instruction=None,
    pooling=None,
    dimension=None,
    trust_remote_code=False,
    vector_cache=None
):
    import time
    if db.row_factory is None:
        db.row_factory = sqlite3.Row
    t_start = time.perf_counter()
    where = []
    params = []
    if source is not None:
        where.append('w.source=?')
        params.append(source)
    if language is not None:
        where.append('(w.language=? OR w.language LIKE ?)')
        params.extend([language, f'%{language}%'])
    if work_id is not None:
        where.append('w.work_id=?')
        params.append(work_id)
    if document_id is not None:
        where.append('w.document_id=?')
        params.append(document_id)

    clause = (' AND ' + ' AND '.join(where)) if where else ''
    lex = []
    dense = []
    k = candidates

    t_lex = 0.0
    t_dense = 0.0
    t_graph = 0.0
    t_rerank = 0.0

    run_lex = mode.startswith('lexical') or mode.startswith('hybrid')
    run_dense = mode.startswith('dense') or mode.startswith('hybrid')
    run_graph = mode.endswith('_graph')
    run_rerank = 'rerank' in mode

    # 1. Lexical retrieval
    if run_lex:
        t0 = time.perf_counter()
        terms = query.split()
        lex_query = '"' + query.replace('"', '""') + '"' if phrase else ' OR '.join('"' + t.replace('"', '""') + '"' for t in terms)
        if lex_query:
            lex = [row['id'] for row in db.execute(
                'SELECT w.id FROM lexical JOIN window w ON w.id=lexical.rowid WHERE lexical MATCH ?' + clause + ' ORDER BY bm25(lexical),w.id LIMIT ?',
                [lex_query, *params, k]
            )]
        t_lex = (time.perf_counter() - t0) * 1000.0

    # 2. Dense retrieval
    if run_dense:
        t0 = time.perf_counter()
        close_enc_at_end = False
        if encoder is None:
            if not model or not revision or not cache:
                raise ValueError('model, revision, and cache are required if encoder is not provided')
            meta = dict(db.execute('SELECT key,value FROM metadata').fetchall())
            requested_model = str(Path(model).resolve()) if Path(model).exists() else model
            if meta.get('model') != requested_model or meta.get('revision') != revision:
                raise ValueError('model identity does not match index')
            encoder = Encoder(model, revision, cache, batch_size=batch,
                              pooling=pooling, dimension=dimension,
                              trust_remote_code=trust_remote_code)
            close_enc_at_end = True

        validate_encoder_compatibility(db, encoder)

        if hasattr(encoder, 'encode_queries'):
            v = encoder.encode_queries([query], task=task, instruction=instruction)[0]
        else:
            v = encoder.encode([query])[0]

        ids, mat = get_vector_matrix(db, clause, tuple(params), cache=vector_cache)
        if len(ids) > 0:
            scores = mat @ v
            if min_dense_sim is not None:
                valid_mask = scores >= min_dense_sim
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    sorted_valid = valid_indices[np.argsort(-scores[valid_indices])[:k]]
                    dense = ids[sorted_valid].tolist()
                else:
                    dense = []
            else:
                top_k_indices = np.argsort(-scores)[:k]
                dense = ids[top_k_indices].tolist()
        else:
            dense = []
        if close_enc_at_end:
            encoder.close()
        t_dense = (time.perf_counter() - t0) * 1000.0

    # 3. Score combination / Reciprocal Rank Fusion
    ranking = {}
    if mode.startswith('hybrid'):
        for route, ids_list in [('lexical', lex), ('dense', dense)]:
            for rank, wid in enumerate(ids_list, 1):
                entry = ranking.setdefault(wid, {'score': 0.0, 'ranks': {}})
                entry['score'] += 1.0 / (rrf + rank)
                entry['ranks'][route] = rank
    elif mode.startswith('lexical'):
        for rank, wid in enumerate(lex, 1):
            ranking[wid] = {'score': 1.0 / (rrf + rank), 'ranks': {'lexical': rank}}
    elif mode.startswith('dense'):
        for rank, wid in enumerate(dense, 1):
            ranking[wid] = {'score': 1.0 / (rrf + rank), 'ranks': {'dense': rank}}

    # 4. Graph expansion and proximity boosting
    graphpaths = {}
    if run_graph:
        t0 = time.perf_counter()
        if not graph_db:
            raise ValueError(f'graph_db is required for {mode} mode')

        top_seeds = list(sorted(ranking, key=lambda x: -ranking[x]['score']))[:graph_seeds]
        for rank_seed, wid in enumerate(top_seeds, 1):
            for row in db.execute('SELECT passage_id FROM mapping WHERE window_id=?', (wid,)):
                for n in neighbors(graph_db, 'passage:' + row[0], 2, graph_limit):
                    if not n['entity'].startswith('passage:'):
                        continue
                    pid = n['entity'][8:]
                    hop = len(n['path'])
                    boost = graph_weight * (1.0 / (rrf + hop * 10))

                    for candidate in db.execute('SELECT DISTINCT w.id FROM window w JOIN mapping m ON m.window_id=w.id WHERE m.passage_id=?' + clause, [pid, *params]):
                        target = candidate[0]
                        if target in ranking:
                            ranking[target]['score'] += boost
                            ranking[target]['ranks']['graph_boost'] = round(boost, 5)
                            if target not in graphpaths:
                                graphpaths[target] = n['path']
                        else:
                            ranking[target] = {
                                'score': boost,
                                'ranks': {'graph_hop': hop, 'seed_rank': rank_seed}
                            }
                            graphpaths[target] = n['path']
        t_graph = (time.perf_counter() - t0) * 1000.0

    seen = set()
    output = []
    fetch_limit = rerank_candidates if run_rerank else limit
    for wid in sorted(ranking, key=lambda x: (-ranking[x]['score'], x)):
        passages = []
        for row in db.execute('SELECT p.*,m.start,m.end FROM mapping m JOIN passage p ON p.id=m.passage_id WHERE m.window_id=? ORDER BY m.position', (wid,)):
            p = dict(row)
            p['record'] = json.loads(p['record'])
            passages.append(p)
        if all(p['id'] in seen for p in passages):
            continue
        seen.update(p['id'] for p in passages)
        w = dict(db.execute('SELECT id,text,source,language,work_id,document_id FROM window WHERE id=?', (wid,)).fetchone())
        output.append({
            'corpus_id': 'ancient-corpus',
            'window': w,
            'passages': passages,
            'retrieval': ranking[wid],
            'graph_evidence': graphpaths.get(wid, [])
        })
        if len(output) >= fetch_limit:
            break

    # 5. Reranking stage if requested
    if run_rerank:
        t0 = time.perf_counter()
        if reranker is not None:
            output = reranker.rerank(query, output, top_k=limit)
        else:
            # Rule-based fallback reranking
            from rerank import Reranker
            fb_reranker = Reranker()
            output = fb_reranker.rerank(query, output, top_k=limit)
        t_rerank = (time.perf_counter() - t0) * 1000.0
    else:
        output = output[:limit]

    total_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        'results': output,
        'total_candidates': len(ranking),
        'timings_ms': {
            'total': total_ms,
            'lexical': t_lex,
            'dense': t_dense,
            'graph': t_graph,
            'rerank': t_rerank
        }
    }

def search(a):
    db = sqlite3.connect(f'file:{Path(a.db).resolve()}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    graph_db = None
    if a.mode == 'hybrid_graph' and a.graph:
        graph_db = sqlite3.connect(f'file:{Path(a.graph).resolve()}?mode=ro', uri=True)

    res = search_unified(
        db=db,
        query=a.query,
        mode=a.mode,
        model=a.model,
        revision=a.revision,
        cache=a.cache,
        graph_db=graph_db,
        graph_seeds=a.graph_seeds,
        graph_limit=a.graph_limit,
        graph_weight=getattr(a, 'graph_weight', 0.1),
        min_dense_sim=getattr(a, 'min_dense_sim', None),
        limit=a.limit,
        candidates=a.candidates,
        rrf=a.rrf,
        phrase=a.phrase,
        source=a.source,
        language=a.language,
        work_id=a.work_id,
        document_id=a.document_id,
        batch=a.batch,
        pooling=a.pooling,
        dimension=a.dimension,
        trust_remote_code=a.trust_remote_code
    )
    if graph_db:
        graph_db.close()
    db.close()
    print(json.dumps(res['results'], ensure_ascii=False))

def inspect(a):
    db=sqlite3.connect(f'file:{Path(a.db).resolve()}?mode=ro',uri=True); db.row_factory=sqlite3.Row
    row=db.execute('SELECT * FROM passage WHERE id=?',(a.passage,)).fetchone()
    if not row: raise ValueError('passage not found')
    result=dict(row); result['record']=json.loads(result['record']); result['windows']=[dict(x) for x in db.execute('SELECT w.id,w.text,m.start,m.end FROM window w JOIN mapping m ON w.id=m.window_id WHERE m.passage_id=?',(a.passage,))]
    print(json.dumps(result,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    b=sub.add_parser('build'); b.add_argument('--selection',required=True); b.add_argument('--db',required=True); b.add_argument('--tokens',type=int,default=128)
    s=sub.add_parser('search'); s.add_argument('--db',required=True); s.add_argument('--mode',choices=['lexical','dense','hybrid','hybrid_graph'],default='hybrid'); s.add_argument('--graph'); s.add_argument('--graph-seeds',type=int,default=3); s.add_argument('--graph-limit',type=int,default=40); s.add_argument('--graph-weight',type=float,default=0.1); s.add_argument('--min-dense-sim',type=float,default=None); s.add_argument('--limit',type=int,default=10); s.add_argument('--candidates',type=int,default=50); s.add_argument('--rrf',type=int,default=60); s.add_argument('--phrase',action='store_true'); s.add_argument('--source'); s.add_argument('--language'); s.add_argument('--work-id'); s.add_argument('--document-id'); s.add_argument('query')
    i=sub.add_parser('inspect'); i.add_argument('--db',required=True); i.add_argument('passage')
    b.add_argument('--model', required=True)
    b.add_argument('--revision', required=True)
    b.add_argument('--cache', required=True)
    s.add_argument('--model')
    s.add_argument('--revision')
    s.add_argument('--cache')
    for x in (b,s):
        x.add_argument('--batch', type=int, default=64)
        x.add_argument('--pooling', choices=['mean', 'last_token', 'cls'])
        x.add_argument('--dimension', type=int)
        x.add_argument('--trust-remote-code', action='store_true',
                       help='Allow custom Python code from the selected model repository')
    a=p.parse_args(); {'build':build,'search':search,'inspect':inspect}[a.cmd](a)
if __name__=='__main__': main()
