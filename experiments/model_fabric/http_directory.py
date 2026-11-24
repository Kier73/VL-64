"""Read a published snapshot directory through standard HTTP byte ranges."""
import re
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class HTTPDirectory:
    authority = "http-directory-range"

    def __init__(self, root):
        """Bind logical file j to root/j; remember its first strong ETag E_j."""
        parts = urlsplit(root)
        if parts.scheme not in ("https", "http") or not parts.hostname or parts.query or parts.fragment:
            raise ValueError("Expected an HTTP snapshot directory")
        self.root = self.model = root.rstrip("/")
        self.revision = None
        self.etags = {}

    def open(self, name, headers):
        """Request one file within the supplied root, constraining each later read to E_j."""
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*",name):
            raise ValueError("Expected a direct snapshot member")
        headers["Accept-Encoding"] = "identity"
        if name in self.etags:
            headers["If-Match"] = self.etags[name]
        return urlopen(Request(self.root+"/"+name,headers=headers),timeout=30)

    def metadata(self, name, limit):
        """Read the metadata document once with |M|≤limit."""
        try:
            with self.open(name,{}) as response:
                data = response.read(limit+1)
                if response.status != 200 or len(data)>limit:
                    raise ValueError("Incomplete or oversized metadata")
                return data
        except HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(name) from error
            raise

    def read_range(self, name, start, length):
        """Admit exactly S_j[a:a+n] under constant E_j and Content-Range=[a,a+n−1]/N."""
        end = start+length-1
        with self.open(name,{"Range":f"bytes={start}-{end}"}) as response:
            span = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",response.headers.get("Content-Range",""))
            tag = response.headers.get("ETag","")
            if (response.status != 206 or not span or tuple(map(int,span.groups()[:2])) != (start,end)
                    or int(span[3])<=end or not re.fullmatch(r'"[^"]*"',tag)
                    or self.etags.setdefault(name,tag) != tag):
                raise ValueError("Source range or strong ETag mismatch")
            data = response.read(length+1)
            if len(data)!=length:
                raise ValueError("Incomplete source range")
            return data,int(span[3])

    def close(self):
        """Open response count = 0 after each request context exits."""
