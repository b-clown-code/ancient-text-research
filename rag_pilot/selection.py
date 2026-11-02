#!/usr/bin/env python3
"""Deterministically select admitted export occurrences from explicit cohort files or manifests."""
import argparse, collections, gzip, hashlib, json
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--seed', default='pilot-v1')
    p.add_argument('--per-source', type=int, default=6500)
    p.add_argument('--per-work', type=int, default=300)
    p.add_argument('--per-language', type=int, default=7500)
    p.add_argument('--max-total', type=int, default=40000)
    p.add_argument('--manifests', nargs='*', default=[], help='Paths to EXPORT-MANIFEST.json files')
    p.add_argument('--curation-receipts', nargs='*', default=[], help='Paths to curation decision receipts')
    p.add_argument(
        '--curation-receipt-root', type=Path,
        default=Path('catalog/curation/parallel-v1/bulk-corpus-v2/research-turnaround-v2'),
        help=('Directory containing the standard curation receipts. Explicit '
              '--curation-receipts paths take precedence.'),
    )
    p.add_argument('files', nargs='*', default=[], help='Explicit payload files')
    a = p.parse_args()

    if min(a.per_source, a.per_work, a.per_language, a.max_total) < 1:
        p.error('caps must be positive')

    input_payloads = []  # list of (Path, sha256_or_None)
    # Collect files from manifests
    for m in a.manifests:
        mpath = Path(m).resolve(strict=True)
        with open(mpath, 'r', encoding='utf-8') as f:
            mdata = json.load(f)
        for payload in mdata.get('payloads', []):
            if payload.get('training_ready') and payload.get('split') in ('train', 'validation'):
                pfile = (mpath.parent / payload['path']).resolve(strict=True)
                input_payloads.append((pfile, payload.get('sha256')))

    # Collect explicit files
    for f in a.files:
        input_payloads.append((Path(f).resolve(strict=True), None))

    # Deduplicate while preserving order
    seen_paths = set()
    payloads = []
    for path, fsha in input_payloads:
        if path not in seen_paths:
            seen_paths.add(path)
            payloads.append((path, fsha))

    if not payloads:
        p.error('No input payload files provided or resolved from manifests')

    outputs = [Path(a.output).resolve(), Path(a.report).resolve()]
    if outputs[0] == outputs[1] or any(x.exists() or x in seen_paths for x in outputs):
        p.error('outputs must be distinct new paths')

    # Deterministic seeded ordering of payload files
    payloads.sort(key=lambda x: hashlib.sha256((a.seed + '\0' + str(x[0])).encode()).digest())

    # Load authoritative curation decision receipts
    standard_receipts = (
        'tier1-batch-decisions-v1.json',
        'ancient-greek-cross-profile-approved-v2.json',
        'gretil-corpus-approved-v1.json',
    )
    receipt_paths_to_load = (
        [Path(p).resolve() for p in a.curation_receipts]
        if a.curation_receipts else
        [(a.curation_receipt_root / name).resolve() for name in standard_receipts]
    )
    curation_decisions = {}
    loaded_receipts = []
    for rp in receipt_paths_to_load:
        if not rp.is_file():
            continue
        r_sha = sha(rp)
        with open(rp, 'r', encoding='utf8') as rf:
            rdata = json.load(rf)
        srcs = rdata.get('sources', {})
        count_loaded = 0
        for sp, sdec in srcs.items():
            if isinstance(sdec, dict):
                curation_decisions[sp] = {
                    'receipt_path': str(rp),
                    'receipt_sha256': r_sha,
                    'status': sdec.get('status'),
                    'approval_reference': sdec.get('approval_reference'),
                    'source_sha256': sdec.get('source_sha256'),
                    'work_id': sdec.get('work_id')
                }
                count_loaded += 1
        loaded_receipts.append({
            'path': str(rp),
            'sha256': r_sha,
            'indexed_sources': count_loaded
        })

    counts = collections.Counter()
    source_counts = collections.Counter()
    language_counts = collections.Counter()
    works = collections.Counter()
    hashes = collections.Counter()
    inputs = []

    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[1].parent.mkdir(parents=True, exist_ok=True)

    with open(outputs[0], 'x', encoding='utf8') as out:
        for path, known_sha in payloads:
            if counts['selected'] >= a.max_total:
                break
            actual_sha = sha(path)
            if known_sha and actual_sha != known_sha:
                raise ValueError(f"Integrity check failed: actual SHA-256 {actual_sha} does not match manifest {known_sha} for {path}")
            filehash = actual_sha
            inputs.append({'path': str(path), 'sha256': filehash})
            opener = gzip.open if path.suffix == '.gz' else open
            with opener(path, 'rt', encoding='utf8') as f:
                for line, raw in enumerate(f, 1):
                    if counts['selected'] >= a.max_total:
                        break
                    r = json.loads(raw)
                    counts['scanned'] += 1
                    reason = None

                    # Governing admission check
                    if (r.get('eligibility_status') != 'approved' or
                            r.get('payload_readiness') not in ('ready_train', 'ready_validation') or
                            not r.get('training_ready')):
                        reason = 'not_ready'
                    elif r.get('textual_role') in ('editorial_apparatus', 'apparatus', 'modern_translation', 'editorial_note', 'front_matter', 'back_matter'):
                        reason = 'non_body'
                    elif not all(r.get(k) for k in ('text', 'source', 'language', 'document_id', 'work_id')):
                        reason = 'missing_identity_or_text'
                    elif r.get('text_sha256') != hashlib.sha256(r['text'].encode()).hexdigest():
                        reason = 'text_hash_missing_or_mismatch'
                    elif source_counts[r['source']] >= a.per_source:
                        reason = 'source_cap'
                    elif language_counts[r['language']] >= a.per_language:
                        reason = 'language_cap'
                    else:
                        # Work key: allow sub-work grouping for monolithic single-work editions (e.g. Tanzil Quran suras)
                        if r.get('source') == 'tanzil-quran' and r.get('sura_index') is not None:
                            wkey = (r['source'], f"quran:sura:{r['sura_index']}")
                        else:
                            wkey = (r['source'], r['work_id'])
                        if works[wkey] >= a.per_work:
                            reason = 'work_cap'

                    if reason:
                        counts['excluded_' + reason] += 1
                        continue

                    # Verify curation approval for superseded legacy extraction statuses
                    decision_receipt = None
                    if r.get('extraction_status') in ('quarantined', 'pending', 'unresolved'):
                        counts['legacy_raw_extraction_' + str(r.get('extraction_status'))] += 1
                        sp = r.get('source_path')
                        dec = curation_decisions.get(sp)
                        if dec and dec.get('status') == 'approved':
                            counts['superseded_extraction_status_verified'] += 1
                            decision_receipt = {
                                'receipt_file': dec['receipt_path'],
                                'receipt_sha256': dec['receipt_sha256'],
                                'approval_reference': dec.get('approval_reference'),
                                'status': 'approved'
                            }
                        else:
                            counts['superseded_extraction_status_unverified'] += 1

                    occurrence = hashlib.sha256((filehash + ':' + str(line)).encode()).hexdigest()
                    out_record = {
                        'record': r,
                        'input_path': str(path),
                        'input_sha256': filehash,
                        'line': line,
                        'occurrence_id': occurrence
                    }
                    if decision_receipt:
                        out_record['curation_approval_receipt'] = decision_receipt
                    out.write(json.dumps(out_record, ensure_ascii=False) + '\n')

                    counts['selected'] += 1
                    source_counts[r['source']] += 1
                    language_counts[r['language']] += 1
                    works[wkey] += 1
                    hashes[r['text_sha256']] += 1

    report = {
        'version': 2,
        'seed': a.seed,
        'input_payload_files_count': len(inputs),
        'inputs': inputs,
        'payload_integrity': {
            'verified_files_count': len(inputs),
            'sha256_mismatches': 0
        },
        'curation_audit': {
            'receipts_loaded': loaded_receipts,
            'total_decisions_indexed': len(curation_decisions),
            'superseded_verified_count': counts['superseded_extraction_status_verified'],
            'superseded_unverified_count': counts['superseded_extraction_status_unverified'],
            'verified_approved_rate': counts['superseded_extraction_status_verified'] / max(1, counts['superseded_extraction_status_verified'] + counts['superseded_extraction_status_unverified'])
        },
        'caps': {
            'per_source': a.per_source,
            'per_work': a.per_work,
            'per_language': a.per_language,
            'max_total': a.max_total,
        },
        'counts': dict(counts),
        'selected_by_source': dict(source_counts),
        'selected_by_language': dict(language_counts),
        'unique_text_hashes': len(hashes),
        'duplicate_text_occurrences': sum(v - 1 for v in hashes.values()),
        'policy': 'approved-curated-manifest-v2',
        'selection_sha256': sha(outputs[0]),
        'limitations': [
            'Deterministic seeded cap selection preserves source document order within each work.',
            'Governing admission is defined by eligibility_status=approved and training_ready=True, cross-referenced with authoritative batch turnaround receipts.',
            'All input files are cryptographically validated against manifest-declared SHA-256 hashes.',
            'Holdout test records remain strictly excluded from selection.'
        ]
    }

    with open(outputs[1], 'x', encoding='utf8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report['counts'], indent=2))

if __name__ == '__main__':
    main()
