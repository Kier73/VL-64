"""Portable, bounded execution from previously admitted trained-weight glances.

Frozen Grounds are externally pinned views of exact host-qualified bytes, not
whole-model signatures or new remote observations. A later observer may request
different byte partitions only when historical observations fully cover them.
No code enters VL-64's compact 64 KiB runtime.
"""
from __future__ import annotations

import bisect
import errno
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from .qualified_range_cache import (_atomic, _canonical, _request, _sha,
                                    MAX_ENTRIES, MAX_RANGE)

SCHEMA = 'vl64.pinned-weight-glance-ground.v1'
HEX = re.compile(r'[0-9a-f]{64}\Z')
MAX_GROUND_JSON_BYTES = 256 * 1024 * 1024


def _admit_head(head):
    if not isinstance(head, str) or HEX.fullmatch(head) is None:
        raise ValueError('An external 64-digit Ground head is required')
    return head


def _verify_page(path, sha, expected_length):
    """Check a source page in 64 KiB windows rather than reading a model tensor."""
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while True:
            block = stream.read(MAX_RANGE)
            if not block:
                break
            size += len(block)
            if size > expected_length:
                raise ValueError('Retained glance exceeds its qualified extent')
            h.update(block)
    if size != expected_length or h.hexdigest() != sha:
        raise ValueError('Retained glance page corrupted or substituted')


def freeze_weight_glances(cache_root, *, expected_cache_head, ground_root,
                          approved_source_hosts):
    """Freeze checked historic ranges as independent content-addressed Ground.

    Caller retains both the source-catalogue pin and returned Ground-head
    outside either tree. Hardlinked pages are always rehashed on replay.
    """
    _admit_head(expected_cache_head)
    if not approved_source_hosts or any(not isinstance(h, str) or not h
                                        for h in approved_source_hosts):
        raise ValueError('The original observer identities require explicit approval')
    cache_dir = Path(cache_root)
    raw = (cache_dir / 'catalogue.json').read_bytes()
    if len(raw) > MAX_GROUND_JSON_BYTES or _sha(raw) != expected_cache_head:
        raise ValueError('External range-cache catalogue head differs')
    try:
        catalogue = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Malformed retained catalogue') from exc
    if (not isinstance(catalogue, dict) or set(catalogue) !=
            {'schema', 'budget_bytes', 'entries', 'model_identity'} or
            catalogue['schema'] != 'vl64.qualified-range-cache.v1' or
            not isinstance(catalogue['entries'], list) or
            not 0 < len(catalogue['entries']) <= MAX_ENTRIES or
            not isinstance(catalogue['model_identity'], list) or
            len(catalogue['model_identity']) != 3):
        raise ValueError('Unqualified source catalogue structure')
    identity = tuple(catalogue['model_identity'])
    if (any(not isinstance(v, str) or not v for v in identity) or
            HEX.fullmatch(identity[2]) is None):
        raise ValueError('Malformed source model identity')
    ground_dir = Path(ground_root)
    ground_dir.mkdir(parents=True, exist_ok=True)
    pages = ground_dir / 'pages'
    pages.mkdir(parents=True, exist_ok=True)
    entries = []
    keys = set()
    for entry in catalogue['entries']:
        if not isinstance(entry, dict) or set(entry) != {'request', 'observer_host', 'sha256'}:
            raise ValueError('Malformed retained witness')
        req = _request(entry['request'])
        source = entry['observer_host']
        sha = _admit_head(entry['sha256'])
        if (req['model'], req['revision'], req['index_ground_digest']) != identity:
            raise ValueError('Observed range crosses model identity')
        if source not in approved_source_hosts:
            raise ValueError('Historical source not approved by new observer')
        key = tuple(req[k] for k in sorted(req))
        if key in keys:
            raise ValueError('Duplicated source-range witness')
        keys.add(key)
        length = req['byte_end_exclusive'] - req['byte_begin']
        original = cache_dir / 'pages' / sha
        _verify_page(original, sha, length)
        target = pages / sha
        if target.exists():
            _verify_page(target, sha, length)
        else:
            try:
                os.link(original, target)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                fd, pending = tempfile.mkstemp(dir=pages, prefix='.pending-')
                try:
                    with os.fdopen(fd, 'wb') as dst, original.open('rb') as src:
                        while True:
                            block = src.read(MAX_RANGE)
                            if not block:
                                break
                            dst.write(block)
                        dst.flush()
                        os.fsync(dst.fileno())
                    _verify_page(Path(pending), sha, length)
                    os.replace(pending, target)
                finally:
                    if os.path.exists(pending):
                        os.unlink(pending)
            _verify_page(target, sha, length)
        entries.append({'request': req, 'observer_host': source, 'sha256': sha})
    body = {'schema': SCHEMA, 'source_catalogue_head': expected_cache_head,
            'model_identity': list(identity), 'entries': entries}
    ground_bytes = _canonical(body)
    if len(ground_bytes) > MAX_GROUND_JSON_BYTES:
        raise ValueError('Frozen Ground manifest exceeds bounded metadata budget')
    target = ground_dir / 'ground.json'
    if target.exists() and target.read_bytes() != ground_bytes:
        raise ValueError('Ground directory already contains different observations')
    _atomic(target, ground_bytes)
    return _sha(ground_bytes)


class PinnedWeightGlanceGround:
    """Offline qualified local observer of retained bytes, with source lineage.

    The local reconstruction identity must be separately approved by the
    QualifiedShardNavigator. It never impersonates the historical HTTP host.
    """

    def __init__(self, root, *, expected_head, approved_source_hosts,
                 local_observer_host='pinned-glance-reconstructor'):
        _admit_head(expected_head)
        if (not isinstance(local_observer_host, str) or not local_observer_host or
                not approved_source_hosts):
            raise ValueError('Ground reader and historical sources must be approved')
        self.root = Path(root)
        raw = (self.root / 'ground.json').read_bytes()
        if len(raw) > MAX_GROUND_JSON_BYTES or _sha(raw) != expected_head:
            raise ValueError('Ground differs from independent externally pinned head')
        try:
            doc = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ValueError('Malformed pinned Ground') from exc
        if (not isinstance(doc, dict) or set(doc) !=
                {'schema', 'source_catalogue_head', 'model_identity', 'entries'} or
                doc['schema'] != SCHEMA or not isinstance(doc['model_identity'], list) or
                len(doc['model_identity']) != 3 or not isinstance(doc['entries'], list) or
                not 0 < len(doc['entries']) <= MAX_ENTRIES):
            raise ValueError('Unadmitted pinned Ground fields')
        _admit_head(doc['source_catalogue_head'])
        self.identity = tuple(doc['model_identity'])
        if (any(not isinstance(x, str) or not x for x in self.identity) or
                HEX.fullmatch(self.identity[2]) is None):
            raise ValueError('Unadmitted pinned model identity')
        self.head = expected_head
        self.source_catalogue_head = doc['source_catalogue_head']
        self.local_observer_host = local_observer_host
        self.approved_sources = frozenset(approved_source_hosts)
        self._intervals = {}
        self._starts = {}
        self.reads = 0
        self.bytes_reconstructed = 0
        for entry in doc['entries']:
            if not isinstance(entry, dict) or set(entry) != {'request', 'observer_host', 'sha256'}:
                raise ValueError('Malformed pinned source witness')
            req = _request(entry['request'])
            if (req['model'], req['revision'], req['index_ground_digest']) != self.identity:
                raise ValueError('Cross-model pinned observation')
            if entry['observer_host'] not in self.approved_sources:
                raise ValueError('Pinned source host no longer authorized')
            _admit_head(entry['sha256'])
            key = (req['shard'], req['purpose'])
            self._intervals.setdefault(key, []).append((
                req['byte_begin'], req['byte_end_exclusive'], entry))
        for key, spans in self._intervals.items():
            spans.sort(key=lambda t: (t[0], t[1]))
            self._starts[key] = [span[0] for span in spans]

    def __call__(self, request):
        req = _request(request)
        if (req['model'], req['revision'], req['index_ground_digest']) != self.identity:
            raise ValueError('Reconstruction model/revision/index differs from pinned Ground')
        lo, hi = req['byte_begin'], req['byte_end_exclusive']
        spans = self._intervals.get((req['shard'], req['purpose']), [])
        starts = self._starts.get((req['shard'], req['purpose']), [])
        first = bisect.bisect_left(starts, max(0, lo - MAX_RANGE + 1))
        last = bisect.bisect_left(starts, hi)
        out = bytearray(hi - lo)
        filled = bytearray(hi - lo)
        origins = []
        for a, b, entry in spans[first:last]:
            begin, end = max(a, lo), min(b, hi)
            if begin >= end:
                continue
            source_page = self.root / 'pages' / entry['sha256']
            _verify_page(source_page, entry['sha256'], b-a)
            with source_page.open('rb') as stream:
                stream.seek(begin-a)
                piece = stream.read(end-begin)
            if len(piece) != end-begin:
                raise ValueError('Incomplete retained glance fragment')
            offset = begin-lo
            for i, value in enumerate(piece):
                k = offset+i
                if filled[k] and out[k] != value:
                    raise ValueError('Contradictory overlapping weight glances')
                out[k], filled[k] = value, 1
            origins.append({'observer_host': entry['observer_host'],
                            'request': entry['request'], 'sha256': entry['sha256'],
                            'used_begin': begin, 'used_end_exclusive': end})
        if not all(filled):
            raise ValueError('UNRESOLVED: unobserved weight gap cannot be reconstructed')
        self.reads += 1
        self.bytes_reconstructed += len(out)
        return {**req, 'status': 'QUALIFIED_EXACT_SHARD_RANGE',
                'observer_host': self.local_observer_host, 'payload': bytes(out),
                'sha256': _sha(out), 'cache_disposition': 'PINNED_GLANCE_DERIVATION',
                'glance_ground_head': self.head, 'original_witnesses': origins}

    def stats(self):
        return {'reconstruction_reads': self.reads,
                'bytes_reconstructed': self.bytes_reconstructed,
                'remote_ranges': 0,
                'scope': 'derived local read from pinned historical exact-byte observations'}
