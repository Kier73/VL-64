import io
import pytest

from experiments.model_fabric.external_transformer import attention, patches
from experiments.model_fabric.http_directory import HTTPDirectory
from experiments.seed_fabric.build_model import build_model


def test_attention_masks_change_future_dependence():
    """For q=k=0, softmax yields the visible arithmetic mean."""
    q = [[0.],[0.]]
    v = [[2.],[8.]]
    assert attention(q,q,v,1,True)==[[2.],[5.]]
    assert attention(q,q,v,1,False)==[[5.],[5.]]


def test_convolution_patch_order_and_stride():
    """A patch at (2,2) spans (1:4,1:4) per channel with zero padding at coordinate 3."""
    planes = [list(range(9)), list(range(10,19))]
    windows = list(patches(planes,3,3,2,0))
    assert len(windows) == 4
    assert windows[-1] == (4,5,0,7,8,0,0,0,0,14,15,0,17,18,0,0,0,0)
    assert list(patches(planes,3,1,2,0)) == [(0,10),(2,12),(6,16),(8,18)]


def test_max_pool_border_keeps_negative_maximum():
    """max(X∪{−∞})=max(X), including when every visible activation is negative."""
    assert max(next(patches([[-3.]],1,3,2,float("-inf")))) == -3.


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
