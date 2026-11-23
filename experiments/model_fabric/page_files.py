"""Navigate published Hub HTML to the immutable operands of a VL-64 computation."""
import hashlib
from contextlib import ExitStack
from html.parser import HTMLParser
import json
import re
from tempfile import TemporaryFile
from urllib.parse import quote

from .hf_glance import _read, _url, range_bytes

SITE = "https://huggingface.co"
PIN = r"[0-9a-f]{40}"


def visit(url, visits):
    """Append (u, |body|, H(body)) to the observed path."""
    raw, receipt = _read(url, limit=4 << 20)
    visits.append(dict(url=url, bytes=len(raw), sha256=receipt["sha256"]))
    return raw, receipt


class Page(HTMLParser):
    def __init__(self, url, visits):
        """Record (u, |body|, H(body)) and extract the outgoing edges E(u)."""
        super().__init__()
        self.links = set()
        self.views = {}
        body, _ = visit(url, visits)
        self.feed(body.decode())
        self.close()

    def handle_starttag(self, tag, attrs):
        """E ← E ∪ {href}; retain the unique typed directory and ref descriptions."""
        attrs = dict(attrs)
        if tag == "a":
            self.links.add(attrs.get("href"))
        view = attrs.get("data-target")
        if view in ("ViewerIndexTreeList", "ViewerHeader"):
            if view in self.views:
                raise ValueError("Ambiguous tree component")
            self.views[view] = json.loads(attrs["data-props"])

    def require(self, path):
        """Follow u → v only if v ∈ E(u)."""
        if path not in self.links:
            raise ValueError("Missing page link: " + path)
        return SITE + path


class PageFiles(ExitStack):
    def __init__(self, entry, search=False):
        """Resolve search → card → tree → commit, restricting files to that commit's links."""
        super().__init__()
        match = re.fullmatch((r"" if search else re.escape(SITE) + "/") +
                             r"([\w.-]+/[\w.-]+)(?:/tree/(main|" + PIN + r"))?/?", entry, re.ASCII)
        if not match or (search and match[2]):
            raise ValueError("Expected a Hub model card, root tree, or exact repository search")
        self.model, rev = match.groups()
        _url(self.model, "0" * 40, "config.json")
        root = "/" + self.model
        visits = []
        if search:
            Page(SITE + "/models?search=" + quote(self.model, safe=""), visits).require(root)
        if rev is None:
            Page(SITE + root, visits).require(root + "/tree/main")
            rev = "main"
        tree = SITE + root + "/tree/"
        page = Page(tree + rev, visits)
        repo = {"name": self.model, "type": "model"}
        if rev == "main":
            head = page.views["ViewerHeader"]
            if head["context"] != {"repo": repo, "rev": rev}:
                raise ValueError("Branch context mismatch")
            pins = [r["targetCommit"] for r in head["refs"]["branches"] if r["ref"] == "refs/heads/main"]
            if len(pins) != 1 or not re.fullmatch(PIN, pins[0]):
                raise ValueError("Missing unique branch tip")
            rev = pins[0]
            page = Page(tree + rev, visits)
        view = page.views["ViewerIndexTreeList"]
        if view["context"] != {"repo": repo, "rev": rev}:
            raise ValueError("Pinned tree context mismatch")
        self.revision = rev
        self.entries = {}
        items = view["entries"]
        first = len(items)
        cursor = view["nextURL"]
        while cursor:
            prefix = "/api/models/" + self.model + "/tree/" + rev + "?"
            if not cursor.startswith(prefix) or len(visits) >= 24:
                raise ValueError("Unqualified directory continuation")
            url = SITE + cursor
            raw, receipt = visit(url, visits)
            items.extend(json.loads(raw))
            link = receipt.get("link")
            match = re.fullmatch(r'<https://huggingface\.co(/[^>]+)>; rel="next"', link) if link else None
            if link and not match:
                raise ValueError("Unknown pagination link")
            cursor = match[1] if match else None
        for i, item in enumerate(items):
            name = item["path"]
            if item["type"] == "file" and (name in ("config.json", "model.safetensors.index.json")
                                           or name.endswith(".safetensors")):
                path = _url(self.model, rev, name)[len(SITE):] + "?download=true"
                if name in self.entries:
                    raise ValueError("Duplicate directory file")
                self.entries[name] = (page.require(path) if i < first else SITE + path, item)
        self.visits = visits
        self.authority = "pinned-huggingface-https-range"
        self.staged = {}
        self.transferred = 0

    def metadata(self, name, limit):
        """Check H('blob ' ‖ |M| ‖ NUL ‖ M) or the LFS SHA-256 identity."""
        if name not in self.entries:
            if name == "model.safetensors.index.json":
                raise FileNotFoundError(name)
            raise ValueError("Missing directory entry: " + name)
        url, item = self.entries[name]
        raw, receipt = _read(url, limit=limit)
        storage = item.get("lfs") or item
        if storage is not item:
            observed = receipt["sha256"]
        else:
            observed = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
        if len(raw) != item["size"] or observed != storage["oid"]:
            raise ValueError("Directory file identity mismatch")
        return raw

    def read_range(self, name, start, length):
        """Return S_j[a:a+n] under an exact byte-extent witness."""
        url, item = self.entries[name]
        if name in self.staged:
            stream = self.staged[name]
            stream.seek(start)
            data = stream.read(length)
            if len(data) != length:
                raise ValueError("Short staged range")
        else:
            data, total = range_bytes(url, start, length)
            if total != item["size"]:
                raise ValueError("Range extent differs from directory")
            self.transferred += length
        return data, item["size"]

    def stage(self, names, budget):
        """Cache shards only when Σ|S_j| ≤ B and SHA256(S_j) equals the listed LFS oid."""
        if sum(self.entries[n][1]["size"] for n in names) > budget:
            raise ValueError("Selected shards exceed staging budget")
        for name in names:
            size = self.entries[name][1]["size"]
            stream = self.enter_context(TemporaryFile())
            digest = hashlib.sha256()
            for start in range(0, size, 8 << 20):
                data, _ = self.read_range(name, start, min(size-start, 8 << 20))
                stream.write(data)
                digest.update(data)
            if digest.hexdigest() != self.entries[name][1]["lfs"]["oid"]:
                raise ValueError("Shard SHA-256 differs from pinned directory")
            self.staged[name] = stream

