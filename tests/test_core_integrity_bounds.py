"""Core audit: canonical hashes and 4 KiB page size cannot admit invalid values."""
import pytest

from vld.micro_kernel import canonical_bytes, domain_digest
from vld.micro_substrates import PageSpace


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_core_canonical_hash_refuses_nonfinite_values(value):
    with pytest.raises(ValueError): canonical_bytes({'invalid':value})
    with pytest.raises(ValueError): domain_digest('vl64.test.v1',{'invalid':[value]})


def test_cow_pages_do_not_grow_or_allow_negative_offsets():
    pages=PageSpace()
    for offset,data in [(4096,b'x'),(4095,b'xy'),(-1,b'y')]:
        with pytest.raises(ValueError): pages.write_page(0,offset,data)
    with pytest.raises(ValueError):pages.write_page(-1,0,b'x')
    with pytest.raises(ValueError):pages.read_page(-1)
    with pytest.raises((TypeError,ValueError)): pages.write_page(0,0,'x')
    assert pages.read_page(0)==bytes(4096)
    pages.write_page(0,4095,b'Q')
    assert len(pages.read_page(0))==4096
    assert pages.read_page(0)[-1:]==b'Q'
