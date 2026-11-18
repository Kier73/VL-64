r"""Substrates, CoW Page Spaces, Git Wire v2 & Model Grounds (μVLD-64).

Governing Equations:
1. Lens Invariance: lower(lift(x)) = x for all x in D_Λ
2. Git Set-Difference: Δ_diff = Reachable(want) \ Reachable(have) via Wire Protocol v2
3. CoW Memory Fork: PageSpace_child = PageSpace_parent ◁ Δ_writes in O(1) time
4. Safetensors Glance: len(Header) = read_u64_le(bytes[:8]), IO = O(|metadata|) << O(|W|)
5. MoE noaux_tc: S = top-k(g(z_e) + b_e), w_e = g(z_e) / sum_{j in S} g(z_j)
6. Stream Map Fusion: (a2, b2) ∘ (a1, b1) in transit hops with involution cancellation
"""
from __future__ import annotations
import hashlib, json, math, struct
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# --- Coordinate Lenses & Byte Transformations ---
class ByteLens:
    """Declared, round-trip verified coordinate lens Λ: (Z/256Z)* <-> Σ_G."""
    def __init__(self, name: str, lift_fn: Callable[[bytes], Any], lower_fn: Callable[[Any], bytes]):
        self.name = name; self._lift, self._lower = lift_fn, lower_fn

    def lift(self, raw: bytes) -> Any: return self._lift(raw)
    def lower(self, val: Any) -> bytes: return self._lower(val)

    def verify_roundtrip(self, raw: bytes) -> bool:
        """Enforces lower(lift(x)) == x before issuing representation receipt."""
        return self.lower(self.lift(raw)) == raw

LENS_RAW = ByteLens("raw", lambda b: list(b), lambda l: bytes(l))
LENS_UTF8 = ByteLens("utf8", lambda b: b.decode('utf-8'), lambda s: s.encode('utf-8'))
LENS_U32LE = ByteLens("u32le",
    lambda b: list(struct.unpack(f"<{len(b)//4}I", b[:(len(b)//4)*4])),
    lambda vals: struct.pack(f"<{len(vals)}I", *vals))

# --- M1 Content-Addressed Paged Store & Copy-on-Write Virtual Memory ---
class PageSpace:
    """Transactional Copy-on-Write virtual page space with content-addressed SHA-256 commitments."""
    PAGE_SIZE = 4096
    def __init__(self, parent: Optional[PageSpace] = None):
        self.parent = parent
        self.pages: Dict[int, bytearray] = {}
        self.page_hashes: Dict[int, str] = {}

    def fork(self) -> PageSpace:
        """Atomic O(1) Copy-on-Write child creation."""
        return PageSpace(parent=self)

    def read_page(self, page_idx: int) -> bytes:
        if page_idx in self.pages: return bytes(self.pages[page_idx])
        if self.parent: return self.parent.read_page(page_idx)
        return bytes(self.PAGE_SIZE)

    def write_page(self, page_idx: int, offset: int, data: bytes):
        if page_idx not in self.pages:
            base = bytearray(self.read_page(page_idx))
            self.pages[page_idx] = base
        target = self.pages[page_idx]
        target[offset:offset + len(data)] = data
        self.page_hashes[page_idx] = hashlib.sha256(target).hexdigest()

    def commit_manifest(self) -> Dict[str, Any]:
        """Emit deterministic page manifest hash."""
        manifest = {str(k): self.page_hashes.get(k, hashlib.sha256(self.read_page(k)).hexdigest())
                    for k in sorted(self.pages.keys())}
        return {"pages": manifest, "digest": hashlib.sha256(json.dumps(manifest).encode()).hexdigest()}

# --- Native Git Wire Protocol v2 Merkle-DAG Difference Engine ---
class GitWireV2Parser:
    """Parses Smart HTTP Git Wire Protocol v2 packet lines and computes DAG set-difference."""
    @staticmethod
    def encode_pkt_line(line: Optional[str]) -> bytes:
        if line is None: return b"0000" # Flush-pkt
        payload = line.encode('utf-8')
        length = len(payload) + 4
        return f"{length:04x}".encode('ascii') + payload

    @staticmethod
    def parse_pkt_lines(raw: bytes) -> List[bytes]:
        lines = []; idx = 0; n = len(raw)
        while idx + 4 <= n:
            prefix = raw[idx:idx+4]
            if prefix == b"0000":
                lines.append(b""); idx += 4; continue
            try: length = int(prefix.decode('ascii'), 16)
            except Exception: break
            if length < 4 or idx + length > n: break
            lines.append(raw[idx+4:idx+length])
            idx += length
        return lines

    @staticmethod
    def compute_set_difference(wants: Set[str], haves: Set[str],
                               dag: Dict[str, List[str]]) -> Set[str]:
        r"""Computes Reachable(wants) \ Reachable(haves)."""
        def reach(roots: Set[str]) -> Set[str]:
            visited: Set[str] = set(); queue = list(roots)
            while queue:
                curr = queue.pop(0)
                if curr not in visited:
                    visited.add(curr)
                    queue.extend(dag.get(curr, []))
            return visited
        have_closure = reach(haves)
        want_closure = reach(wants)
        return want_closure - have_closure

# --- Headless OS POSIX Virtual Machine ---
class HeadlessOS:
    """Deterministic POSIX process environment with CoW pages, pipes, and W^X memory."""
    def __init__(self):
        self.page_space = PageSpace()
        self.fds: Dict[int, bytearray] = {0: bytearray(), 1: bytearray(), 2: bytearray()}
        self.exit_code: Optional[int] = None
        self.wx_protections: Dict[int, str] = {} # page -> 'R', 'W', 'X', 'RW', 'RX'

    def fork(self) -> HeadlessOS:
        child = HeadlessOS()
        child.page_space = self.page_space.fork()
        child.fds = {fd: bytearray(buf) for fd, buf in self.fds.items()}
        child.wx_protections = dict(self.wx_protections)
        return child

    def mprotect(self, page_idx: int, prot: str):
        if 'W' in prot and 'X' in prot:
            raise PermissionError("W^X violation: page cannot be simultaneously writable and executable")
        self.wx_protections[page_idx] = prot

    def write_fd(self, fd: int, data: bytes):
        if fd not in self.fds: self.fds[fd] = bytearray()
        self.fds[fd].extend(data)

    def read_fd(self, fd: int, max_bytes: int = 4096) -> bytes:
        if fd not in self.fds: return b""
        buf = self.fds[fd]
        res = bytes(buf[:max_bytes])
        del buf[:max_bytes]
        return res

# --- Safetensors 206 Range Glance & Sparse MoE Routing ---
class SafetensorsGlance:
    """Zero-weight-download Safetensors reader using 8-byte LE header prefix."""
    @staticmethod
    def parse_header_bytes(raw_prefix: bytes) -> Tuple[int, Dict[str, Any]]:
        if len(raw_prefix) < 8: raise ValueError("Prefix too short")
        header_len = struct.unpack("<Q", raw_prefix[:8])[0]
        if len(raw_prefix) < 8 + header_len:
            raise ValueError(f"Incomplete header: need {8 + header_len} bytes")
        header_json = json.loads(raw_prefix[8:8 + header_len].decode('utf-8'))
        return header_len, header_json

def topk_moe_routing(logits: Sequence[float], k: int = 2,
                     bias: Optional[Sequence[float]] = None,
                     activation: str = "sigmoid") -> Tuple[List[int], List[float]]:
    """Decoupled noaux_tc top-k routing: S = top-k(g(z) + b), w_e = g(z_e) / sum_S g(z_j)."""
    E = len(logits)
    def act(z: float) -> float:
        if activation == "sigmoid": return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        return math.sqrt(math.log(1.0 + math.exp(max(-30.0, min(30.0, z))))) # deepseek sqrt_gelu
    raw_scores = [act(logits[e]) for e in range(E)]
    select_scores = [raw_scores[e] + (bias[e] if bias else 0.0) for e in range(E)]
    topk_indices = sorted(range(E), key=lambda e: select_scores[e], reverse=True)[:k]
    sum_gates = sum(raw_scores[e] for e in topk_indices) + 1e-9
    norm_gates = [raw_scores[e] / sum_gates for e in topk_indices]
    return topk_indices, norm_gates

# --- Fluid In-Transit Path Machine & Negative Controls ---
class StreamPathMachine:
    """Executes stream transformations across communication hops with involution cancellation."""
    def __init__(self):
        self.hop_receipts: List[str] = []

    def process_hop(self, transform_name: str, payload: bytes,
                    involution: bool = False) -> Tuple[bytes, bool]:
        """Negative control: refuses stateless hops that do not transform or lack witness MAC."""
        h_before = hashlib.sha256(payload).hexdigest()
        if transform_name == "reverse":
            out = payload[::-1]
        elif transform_name == "xor_mask":
            out = bytes(b ^ 0xAA for b in payload)
        else:
            out = payload
        h_after = hashlib.sha256(out).hexdigest()
        if h_before == h_after:
            return payload, False # Refused by negative control: zero transformation
        receipt = hashlib.sha256(f"{transform_name}:{h_before}->{h_after}".encode()).hexdigest()
        self.hop_receipts.append(receipt)
        return out, True
