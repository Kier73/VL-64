import io
import pytest

from experiments.model_fabric.external_transformer import attention
from experiments.model_fabric.http_directory import HTTPDirectory
from experiments.seed_fabric.build_model import build_model


def test_attention_masks_change_future_dependence():
    """For q=k=0, softmax yields the visible arithmetic mean."""
    q = [[0.],[0.]]
    v = [[2.],[8.]]
    assert attention(q,q,v,1,True)==[[2.],[5.]]
    assert attention(q,q,v,1,False)==[[5.],[5.]]


def test_http_directory_binds_ranges_and_version(monkeypatch):
    """Read(A,a,n)=A[a:a+n] only under invariant shard identity E."""
    files = HTTPDirectory("https://example.invalid/pinned")
    tags = iter(['"v1"','"v1"','"v2"'])

    def serve(request,timeout):
        """Return A[3:5] under the sequence of test version witnesses."""
        assert request.headers["Range"]=="bytes=3-4"
        if files.etags:
            assert request.get_header("If-match")=='"v1"'
        stream = io.BytesIO(b"ab")
        stream.status = 206
        stream.headers = {"Content-Range":"bytes 3-4/10","ETag":next(tags)}
        return stream

    monkeypatch.setattr("experiments.model_fabric.http_directory.urlopen",serve)
    for _ in range(2):
        assert files.read_range("model.safetensors",3,2)==(b"ab",10)
    with pytest.raises(ValueError,match="ETag"):
        files.read_range("model.safetensors",3,2)


def test_external_seed_reproducible():
    """SHA256(seed(S)) is deterministic; both archive and expanded code fit 2¹⁶ bytes."""
    blob, _, report = build_model(external=True)
    assert blob == build_model(external=True)[0]
    assert max(len(blob), report["expanded_python_bytes"]) <= 65536
