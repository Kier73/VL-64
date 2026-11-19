"""Bounded, persisted exact-byte cache for qualified model shard observations.

This layer *does not* manufacture host attestations: hits reuse the original
validated range witness (and mark that it is a cache hit). Restores require
an independent, caller-held manifest digest, not a hash copied from disk.
Uncommitted new observations are not durable recovery checkpoints.

The wrapper accepts the existing QualifiedShardNavigator request/response
contract, so the Qwen, DeepSeek, and future operator families use one cache.
No code is added to the 64-KiB VL64 runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path

SCHEMA = 'vl64.qualified-range-cache.v1'
REQUEST_KEYS = frozenset({
    'model', 'revision', 'index_ground_digest', 'shard',
    'byte_begin', 'byte_end_exclusive', 'purpose',
})
HEX = re.compile(r'[0-9a-f]{64}\Z')
MAX_RANGE = 65_536
MAX_ENTRIES = 1_000_000


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(b'vl64.qualified-range-cache.v1\0' + _canonical(value))


def _atomic(path, raw):
    fd, pending = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(pending, path)
        # Directory flush is available on POSIX, but is not universally
        # implemented. The external checkpoint pin remains mandatory.
        if os.name == 'posix':
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def _request(request):
    if not isinstance(request, dict) or set(request) != REQUEST_KEYS:
        raise ValueError('Unadmitted range request fields')
    if any(not isinstance(request[k], str) or not request[k] for k in
           ('model', 'revision', 'index_ground_digest', 'shard', 'purpose')):
        raise ValueError('Unadmitted range identity')
    if not HEX.fullmatch(request['index_ground_digest']):
        raise ValueError('Unadmitted index identity')
    lo, hi = request['byte_begin'], request['byte_end_exclusive']
    if type(lo) is not int or type(hi) is not int or lo < 0 or not lo < hi <= lo + MAX_RANGE:
        raise ValueError('Unbounded or malformed byte range')
    if request['purpose'] not in ('header', 'tensor'):
        raise ValueError('Unadmitted range purpose')
    return dict(request)


class QualifiedRangeCache:
    """Single-writer, size-bounded content-addressed cache for exact shard ranges.

    ``range_host`` must implement the original pinned shard reader. The caller
    supplies an approved host set independently. The original source identity
    travels in every hit receipt; the cache is NOT a new network observer.

    ``snapshot()`` atomically commits a cache catalogue and returns a digest
    that the caller must retain out-of-band. A new process may call
    ``__init__(expected_head=...)`` to read a committed catalogue. A missing
    or altered external head is rejected; no implicit trust in on-disk JSON.
    """

    def __init__(self, root, range_host, *, approved_hosts, budget_bytes=512 * 1024 * 1024,
                 expected_head=None):
        if not callable(range_host) or not approved_hosts or any(
                not isinstance(x, str) or not x for x in approved_hosts):
            raise ValueError('Qualified range provider and host approval required')
        if type(budget_bytes) is not int or not MAX_RANGE <= budget_bytes <= 1 << 40:
            raise ValueError('Cache budget must support a bounded range')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pages = self.root / 'pages'
        self.pages.mkdir(parents=True, exist_ok=True)
        self.catalogue_path = self.root / 'catalogue.json'
        self.host = range_host
        self.approved_hosts = frozenset(approved_hosts)
        self.budget_bytes = budget_bytes
        self._entries = OrderedDict()
        self._model_identity = None
        self.stored_bytes = 0
        self.remote_bytes = 0
        self.remote_ranges = 0
        self.hits = 0
        self.requalifications = 0
        self.evictions = 0
        if self.catalogue_path.exists():
            if not isinstance(expected_head, str) or not HEX.fullmatch(expected_head):
                raise ValueError('Existing range catalogue requires externally pinned head')
            raw = self.catalogue_path.read_bytes()
            if len(raw) > 256 * 1024 * 1024:
                raise ValueError('Unbounded range catalogue')
            if _sha(raw) != expected_head:
                raise ValueError('Range catalogue diverges from external head')
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise ValueError('Malformed range catalogue') from exc
            if (not isinstance(data, dict) or set(data) != {'schema', 'budget_bytes', 'entries', 'model_identity'}
                    or data['schema'] != SCHEMA or data['budget_bytes'] != budget_bytes
                    or not isinstance(data['entries'], list)
                    or not isinstance(data['model_identity'], list)
                    or len(data['model_identity']) not in (0, 3)
                    or len(data['entries']) > MAX_ENTRIES):
                raise ValueError('Unadmitted range catalogue')
            self._model_identity = tuple(data['model_identity']) or None
            for e in data['entries']:
                if not isinstance(e, dict) or set(e) != {'request', 'sha256', 'observer_host'}:
                    raise ValueError('Unadmitted cache entry')
                req = _request(e['request'])
                sha = e['sha256']
                observer = e['observer_host']
                if not isinstance(sha, str) or not HEX.fullmatch(sha) or not isinstance(observer, str) or not observer:
                    raise ValueError('Unqualified saved witness')
                if (self._model_identity is None or (req['model'], req['revision'],
                       req['index_ground_digest']) != self._model_identity):
                    raise ValueError('Saved model identity differs from cache pin')
                key = _digest(req)
                if key in self._entries:
                    raise ValueError('Duplicate cached range')
                self._entries[key] = dict(e)
                self.stored_bytes += req['byte_end_exclusive'] - req['byte_begin']
            if self.stored_bytes > budget_bytes:
                raise ValueError('Persisted range cache exceeds byte budget')
        elif expected_head is not None:
            raise ValueError('Expected retained range catalogue absent')

    def _validate_reply(self, req, reply):
        if not isinstance(reply, dict) or reply.get('status') != 'QUALIFIED_EXACT_SHARD_RANGE':
            raise ValueError('Unqualified model host range response')
        if any(reply.get(k) != v for k, v in req.items()):
            raise ValueError('Host response not bound to exact model range')
        if reply.get('observer_host') not in self.approved_hosts:
            raise ValueError('Observer-host lacks explicit admission')
        raw = reply.get('payload')
        if (not isinstance(raw, bytes) or len(raw) != req['byte_end_exclusive'] - req['byte_begin']
                or reply.get('sha256') != _sha(raw)):
            raise ValueError('Incomplete/corrupted observed model bytes')
        return raw

    def _fetch(self, request):
        reply = self.host(dict(request))
        raw = self._validate_reply(request, reply)
        self.remote_ranges += 1
        self.remote_bytes += len(raw)
        return raw, reply['observer_host']

    def _page(self, sha):
        return self.pages / sha

    def _read_page(self, entry):
        raw = self._page(entry['sha256']).read_bytes()
        req = entry['request']
        if (len(raw) != req['byte_end_exclusive'] - req['byte_begin']
                or _sha(raw) != entry['sha256']):
            raise ValueError('Cached range page corrupt; explicit refusal')
        return raw

    def __call__(self, request):
        req = _request(request)
        identity = (req['model'], req['revision'], req['index_ground_digest'])
        if self._model_identity is not None and identity != self._model_identity:
            raise ValueError('Model identity differs from cache pin')
        key = _digest(req)
        entry = self._entries.get(key)
        if entry is not None:
            original = self._read_page(entry)
            # An offline source's witness may be retained as *historical*
            # evidence only while it remains explicitly approved. Otherwise
            # requalify against a current host, refusing divergent content.
            if entry['observer_host'] in self.approved_hosts:
                self.hits += 1
                self._entries.move_to_end(key)
                return {**req, 'status': 'QUALIFIED_EXACT_SHARD_RANGE',
                        'observer_host': entry['observer_host'],
                        'payload': original, 'sha256': entry['sha256'],
                        'cache_disposition': 'RETAINED_ORIGINAL_OBSERVATION'}
            raw, host_id = self._fetch(req)
            if raw != original:
                raise ValueError('Rehomed range conflicts with externally pinned bytes')
            entry['observer_host'] = host_id
            self.requalifications += 1
            self._entries.move_to_end(key)
            return {**req, 'status': 'QUALIFIED_EXACT_SHARD_RANGE',
                    'observer_host': host_id, 'payload': raw,
                    'sha256': entry['sha256'], 'cache_disposition': 'REQUALIFIED_EQUAL_BYTES'}
        raw, host_id = self._fetch(req)
        if self._model_identity is None:
            self._model_identity = identity
        if len(raw) > self.budget_bytes:
            raise ValueError('Range exceeds cache budget')
        # Evict only entire *logical* ranges. Shared content hashes are kept
        # if still reachable from any retained range entry.
        while self.stored_bytes + len(raw) > self.budget_bytes:
            _, victim = self._entries.popitem(last=False)
            self.stored_bytes -= victim['request']['byte_end_exclusive'] - victim['request']['byte_begin']
            self.evictions += 1
            if not any(item['sha256'] == victim['sha256'] for item in self._entries.values()):
                target = self._page(victim['sha256'])
                if target.exists():
                    target.unlink()
        page = self._page(_sha(raw))
        if page.exists():
            if page.read_bytes() != raw:
                raise ValueError('Existing content-addressed page corrupted')
        else:
            _atomic(page, raw)
        self._entries[key] = {'request': req, 'sha256': _sha(raw), 'observer_host': host_id}
        self.stored_bytes += len(raw)
        return {**req, 'status': 'QUALIFIED_EXACT_SHARD_RANGE',
                'observer_host': host_id, 'payload': raw,
                'sha256': _sha(raw), 'cache_disposition': 'FRESH_HOST_OBSERVATION'}

    def snapshot(self):
        """Commit the current catalogue; return its byte digest for external pinning."""
        body = {'schema': SCHEMA, 'budget_bytes': self.budget_bytes,
                'model_identity': list(self._model_identity or ()),
                'entries': list(self._entries.values())}
        raw = _canonical(body)
        _atomic(self.catalogue_path, raw)
        return _sha(raw)

    def stats(self):
        return {'qualified_remote_ranges': self.remote_ranges,
                'qualified_remote_bytes': self.remote_bytes,
                'retained_cache_hits': self.hits,
                'requalified_equal_byte_ranges': self.requalifications,
                'evicted_logical_ranges': self.evictions,
                'retained_logical_bytes': self.stored_bytes,
                'retained_ranges': len(self._entries),
                'scope': 'original host witnesses retained; cache hits are not new network observations'}
