"""Check the search, mutable ref, pinned directory, and payload boundaries independently."""
import hashlib
import html
import json

import pytest

from experiments.model_fabric import page_files as web


MODEL = "owner/checkpoint"
REV = "a" * 40
ROOT = "/" + MODEL
TREE = ROOT + "/tree/"
WEIGHTS = b"abcd"


def component(name, props):
    """HTML attribute parsing inverts escaping: parse(escape(JSON(x))) = x."""
    return f'<div data-target="{name}" data-props="{html.escape(json.dumps(props))}"></div>'


def website(monkeypatch, *, search_link=ROOT, target=REV, next_path=None, omit_link=False):
    """Build S → C → T_main → T_r with one immutable byte object W_r."""
    repo = dict(name=MODEL, type="model")
    cfg = b'{"model_type":"llama"}'
    config = dict(type="file", path="config.json", size=len(cfg),
                  oid=hashlib.sha1(b"blob %d\0" % len(cfg) + cfg).hexdigest())
    shard = dict(type="file", path="model.safetensors", size=len(WEIGHTS),
                 lfs=dict(oid=hashlib.sha256(WEIGHTS).hexdigest()))
    path = ROOT + "/resolve/" + REV + "/"
    entries = [config] if next_path else [config, shard]
    links = "".join(f'<a href="{path}{i["path"]}?download=true"></a>' for i in entries)
    documents = {
        "/models?search=owner%2Fcheckpoint": f'<a href="{search_link}"></a>',
        ROOT: f'<a href="{TREE}main"></a>',
        TREE + "main": component("ViewerHeader", dict(context=dict(repo=repo, rev="main"),
            refs=dict(branches=[dict(ref="refs/heads/main", targetCommit=REV)]))),
        TREE + REV: component("ViewerIndexTreeList", dict(context=dict(repo=repo, rev=target),
            entries=entries, nextURL=next_path)) + ("" if omit_link else links),
        path + "config.json?download=true": cfg.decode(),
    }
    if next_path:
        documents[next_path] = json.dumps([shard])

    def fetch(url, *, limit):
        """Return exactly document[u] when |document[u]| ≤ limit."""
        assert url.startswith(web.SITE + "/")
        raw = documents[url[len(web.SITE):]].encode()
        assert len(raw) <= limit
        return raw, {"sha256": hashlib.sha256(raw).hexdigest()}

    def read(url, start, length):
        """Return (W[a:a+n], |W|) only for the directory's pinned shard."""
        assert url == web.SITE + path + "model.safetensors?download=true"
        return WEIGHTS[start:start+length], len(WEIGHTS)

    monkeypatch.setattr(web, "_read", fetch)
    monkeypatch.setattr(web, "range_bytes", read)
    return documents


@pytest.mark.parametrize("paged", [False, True])
def test_discover_and_stage(monkeypatch, paged):
    """Range(stage(W),a,n) = W[a:a+n]; transfer cost after staging is constant."""
    cursor = "/api/models/" + MODEL + "/tree/" + REV + "?cursor=2" if paged else None
    website(monkeypatch, next_path=cursor)
    with web.PageFiles(MODEL, search=True) as source:
        assert source.revision == REV and len(source.visits) == 4 + paged
        assert source.metadata("config.json", 100) == b'{"model_type":"llama"}'
        with pytest.raises(FileNotFoundError):
            source.metadata("model.safetensors.index.json", 100)
        with pytest.raises(ValueError, match="budget"):
            source.stage(["model.safetensors"], 3)
        source.stage(["model.safetensors"], 4)
        assert source.read_range("model.safetensors", 1, 2) == (b"bc", 4)
        assert source.transferred == 4
        stream = source.staged["model.safetensors"]
    assert stream.closed


@pytest.mark.parametrize("changes", [
    {"search_link": ROOT + "-different"}, {"target": "b" * 40}, {"omit_link": True},
    {"next_path": "/api/models/other/model/tree/" + REV + "?cursor=2"},
])
def test_broken_navigation(monkeypatch, changes):
    """Any missing edge or disagreement in (repo, revision) invalidates the path."""
    website(monkeypatch, **changes)
    with pytest.raises(ValueError):
        web.PageFiles(MODEL, search=True)


def test_changed_bytes(monkeypatch):
    """If H(B') ≠ H(B), neither config(B') nor stage(B') is admitted."""
    documents = website(monkeypatch)
    with web.PageFiles(web.SITE + TREE + REV) as source:
        docs_key = source.entries["config.json"][0][len(web.SITE):]
        documents[docs_key] = documents[docs_key].replace("llama", "other")
        with pytest.raises(ValueError, match="identity"):
            source.metadata("config.json", 100)
        source.entries["model.safetensors"][1]["lfs"]["oid"] = "0" * 64
        with pytest.raises(ValueError, match="SHA-256"):
            source.stage(["model.safetensors"], 4)
        assert not source.staged
