"""Algebraic Linear Operators, Neural Codecs & Precision Verifiers (μVLD-64).

Governing Equations:
1. Sufficient-Statistic Contraction: y_i = (α+β i)S0 + (γ+δ i)S1 + sum_q c_{(q - s i) mod P} x~_q + U (V^T x) + Δ x
2. Codecs: FP8_E4M3(u) = (-1)^s * 2^(e-7) * (1 + m/8), UE8M0(u) = 2^(u - 127)
3. Sub-stochastic Sinkhorn: C^(t)_{i,j} = C~_{i,j} / (sum_r C~_{r,j} + ε) => ||C||_1 < 1
4. Flash-Stream Monoid: (mA, lA, OA) ⊗ (mB, lB, OB) = (mAB, e^(mA-mAB) lA + e^(mB-mAB) lB, e^(mA-mAB) OA + e^(mB-mAB) OB)
5. Double-Centered CKA: CKA(G1, G2) = <H G1 H, H G2 H>_F / (||H G1 H||_F ||H G2 H||_F)
6. Precision Margin Certificate: (z~_1 - z~_v) - 2 ||w_1 - w_v||_2 * ε_terminal - 2 η_head > 0
7. SNC Polyhedral Hinge: f(x) = c + sum A_i (ReLU(x_i) - ReLU(-x_i)) + sum γ_k ReLU(<n_k, x> + β_k)
8. M1 LoRA Morphism: W' = W + (α / r) A B^T
"""
from __future__ import annotations
import math, struct
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- Sufficient-Statistic Linear Operator Contraction ---
class StructuralLinearOperator:
    """Evaluates W x = (W_bilinear + W_orbit + U V^T + Δ) x in O(M + N + M P + R(M+N) + |supp(Δ)|)."""
    def __init__(self, rows: int, cols: int, alpha: float = 0.0, beta: float = 0.0,
                 gamma: float = 0.0, delta: float = 0.0, orbit: Optional[Sequence[float]] = None,
                 stride: int = 1, U: Optional[Sequence[Sequence[float]]] = None,
                 V: Optional[Sequence[Sequence[float]]] = None,
                 sparse_delta: Optional[Dict[Tuple[int, int], float]] = None):
        self.rows, self.cols = rows, cols
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.orbit = list(orbit) if orbit else []
        self.period = len(self.orbit)
        self.stride = stride
        self.U = [list(r) for r in U] if U else []
        self.V = [list(r) for r in V] if V else []
        self.rank = len(self.U[0]) if self.U else 0
        self.sparse_delta = sparse_delta or {}

    def matvec(self, x: Sequence[float]) -> List[float]:
        if len(x) != self.cols: raise ValueError("Dimension mismatch")
        s0 = sum(x)
        s1 = sum(j * xj for j, xj in enumerate(x))
        folded = [0.0] * self.period
        if self.period > 0:
            for j, xj in enumerate(x): folded[j % self.period] += xj
        # Low-rank inner products: z_r = V_{:,r}^T x
        z_rank = [0.0] * self.rank
        for r in range(self.rank):
            z_rank[r] = sum(self.V[j][r] * x[j] for j in range(self.cols))
        out = [0.0] * self.rows
        for i in range(self.rows):
            acc = (self.alpha + self.beta * i) * s0 + (self.gamma + self.delta * i) * s1
            if self.period > 0:
                shift = (self.stride * i) % self.period
                for q in range(self.period):
                    idx = (q - shift) % self.period
                    acc += self.orbit[idx] * folded[q]
            if self.rank > 0:
                acc += sum(self.U[i][r] * z_rank[r] for r in range(self.rank))
            out[i] = acc
        for (r, c), val in self.sparse_delta.items():
            if 0 <= r < self.rows and 0 <= c < self.cols:
                out[r] += val * x[c]
        return out

# --- Block-Scaled Tensor Codecs: FP8 (E4M3), UE8M0, MXFP4 (E2M1) ---
def dequant_e4m3_byte(u: int) -> float:
    """Dequantize OCP FP8 E4M3 byte."""
    if u in (0x7F, 0xFF): return float('nan')
    s = -1.0 if (u & 0x80) else 1.0
    e = (u >> 3) & 0x0F
    m = u & 0x07
    if e == 0: return s * (2.0 ** -6) * (m / 8.0)
    return s * (2.0 ** (e - 7)) * (1.0 + m / 8.0)

def dequant_ue8m0_byte(u: int) -> float:
    """Dequantize UE8M0 scale byte: 2^(u - 127)."""
    if u == 0xFF: return float('nan')
    return 2.0 ** (u - 127)

def dequant_mxfp4_nibble(nib: int) -> float:
    """Dequantize E2M1 FP4 nibble (0..15)."""
    s = -1.0 if (nib & 0x08) else 1.0
    e = (nib >> 1) & 0x03
    m = nib & 0x01
    if e == 0: return s * 0.5 * (m / 2.0)
    return s * (2.0 ** (e - 1)) * (1.0 + m / 2.0)

def dequant_block_fp8(q_bytes: bytes, scale_bytes: bytes, rows: int, cols: int,
                      block_r: int = 32, block_c: int = 32) -> List[List[float]]:
    """Dequantize 2D matrix scaled by block UE8M0 exponents."""
    out = [[0.0] * cols for _ in range(rows)]
    scale_cols = (cols + block_c - 1) // block_c
    for r in range(rows):
        br = r // block_r
        for c in range(cols):
            bc = c // block_c
            scale = dequant_ue8m0_byte(scale_bytes[br * scale_cols + bc])
            val = dequant_e4m3_byte(q_bytes[r * cols + c])
            out[r][c] = val * scale
    return out

# --- Sinkhorn-Knopp & RMSNorm ---
def sinkhorn_birkhoff_substochastic(matrix: Sequence[Sequence[float]], iterations: int = 20,
                                    eps: float = 1e-6) -> List[List[float]]:
    """Row/column alternating Sinkhorn normalization terminating on column substochastic division."""
    M = len(matrix); N = len(matrix[0])
    C = [[math.exp(val) + eps for val in row] for row in matrix]
    for _ in range(iterations):
        # Row normalization
        for i in range(M):
            row_sum = sum(C[i]) + eps
            for j in range(N): C[i][j] /= row_sum
        # Column normalization with strict substochastic bound: sum_i C[i][j] = S_j / (S_j + eps) < 1
        for j in range(N):
            col_sum = sum(C[i][j] for i in range(M)) + eps
            for i in range(M): C[i][j] /= col_sum
    return C

def rms_norm(x: Sequence[float], weight: Sequence[float], eps: float = 1e-6) -> List[float]:
    """Row-wise FP32 RMSNorm."""
    d = len(x)
    mean_sq = sum(xi * xi for xi in x) / d
    scale = 1.0 / math.sqrt(mean_sq + eps)
    return [xi * scale * wi for xi, wi in zip(x, weight)]

# --- Flash-Stream Online Attention Monoid ---
class FlashStreamState:
    """Exact associative monoid (m, l, O) for multi-shard streaming attention."""
    def __init__(self, m: float = -float('inf'), l: float = 0.0, O: Optional[Sequence[float]] = None):
        self.m = m; self.l = l
        self.O = list(O) if O else []

    def combine(self, other: FlashStreamState) -> FlashStreamState:
        if self.l == 0.0: return other
        if other.l == 0.0: return self
        m_ab = max(self.m, other.m)
        alpha = math.exp(self.m - m_ab)
        beta = math.exp(other.m - m_ab)
        l_ab = alpha * self.l + beta * other.l
        d = max(len(self.O), len(other.O))
        O_ab = [alpha * self.O[i] + beta * other.O[i] for i in range(d)]
        return FlashStreamState(m_ab, l_ab, O_ab)

    def normalize(self) -> List[float]:
        if self.l <= 0.0: return [0.0] * len(self.O)
        return [oi / self.l for oi in self.O]

# --- Double-Centered Token-Gram CKA Cross-Model Consensus ---
def linear_cka_consensus(trajectories_A: Sequence[Sequence[float]],
                         trajectories_B: Sequence[Sequence[float]]) -> Tuple[float, bool]:
    """Compute double-centered Linear CKA across pooled trajectories of M statements."""
    M = len(trajectories_A)
    if M != len(trajectories_B) or M < 2: return 0.0, False
    # Gram matrices G = X X^T
    GA = [[sum(trajectories_A[i][k] * trajectories_A[j][k] for k in range(len(trajectories_A[0])))
           for j in range(M)] for i in range(M)]
    GB = [[sum(trajectories_B[i][k] * trajectories_B[j][k] for k in range(len(trajectories_B[0])))
           for j in range(M)] for i in range(M)]
    # Double-centering: H G H with H = I - 1/M 1 1^T
    def double_center(G: List[List[float]]) -> List[List[float]]:
        row_means = [sum(row)/M for row in G]
        col_means = [sum(G[i][j] for i in range(M))/M for j in range(M)]
        total_mean = sum(row_means)/M
        return [[G[i][j] - row_means[i] - col_means[j] + total_mean for j in range(M)] for i in range(M)]
    HGAH = double_center(GA); HGBH = double_center(GB)
    frob_dot = sum(HGAH[i][j] * HGBH[i][j] for i in range(M) for j in range(M))
    normA = math.sqrt(sum(HGAH[i][j]**2 for i in range(M) for j in range(M)))
    normB = math.sqrt(sum(HGBH[i][j]**2 for i in range(M) for j in range(M)))
    if normA == 0.0 or normB == 0.0: return 0.0, False
    cka = frob_dot / (normA * normB)
    return cka, cka >= 0.85

# --- Proof-Carrying Precision Margin Certificate (Fixes Defect 1) ---
class PrecisionCertificate:
    """Exact rational error enclosure and certified logit-margin invariance verifier."""
    @staticmethod
    def verify_margin_invariance(logits: Sequence[float], head_weights: Sequence[Sequence[float]],
                                 terminal_error_radius: Fraction, head_rounding_eps: Fraction,
                                 claimed_verdict: bool) -> bool:
        """Enforces: (z_1 - z_v) >= 2 ||w_1 - w_v||_2 * ε_terminal + 2 η_head. Unsupported claims fail closed."""
        v_top = max(range(len(logits)), key=lambda i: logits[i])
        z_top = Fraction(logits[v_top])
        w_top = head_weights[v_top]
        for v in range(len(logits)):
            if v == v_top: continue
            z_v = Fraction(logits[v])
            margin = z_top - z_v
            diff_norm_sq = sum((w_top[k] - head_weights[v][k])**2 for k in range(len(w_top)))
            diff_l2 = Fraction(math.isqrt(int(diff_norm_sq * 10**12)) + 1, 10**6)
            min_perturbation = 2 * diff_l2 * terminal_error_radius + 2 * head_rounding_eps
            if margin <= min_perturbation:
                return False if claimed_verdict else True
        return True if claimed_verdict else False

# --- Constructive Semantic-to-Neural Hinge Synthesis (SNC) ---
def compile_polyhedral_hinge_1d(pieces: Sequence[Tuple[Fraction, Fraction, Fraction]]) -> Dict[str, Any]:
    """Compiles 1D piecewise-affine function into exact 2-layer rational ReLU network."""
    # pieces: list of (x_break, slope, bias)
    c0 = pieces[0][2]
    hidden_ridges = []
    prev_slope = pieces[0][1]
    for x_b, slope, _ in pieces[1:]:
        delta_slope = slope - prev_slope
        hidden_ridges.append({"slope": delta_slope, "x_break": x_b})
        prev_slope = slope
    return {"c0": c0, "ridges": hidden_ridges}

def eval_polyhedral_hinge_1d(hinge: Dict[str, Any], x: Fraction) -> Fraction:
    res = hinge["c0"]
    for r in hinge["ridges"]:
        diff = x - r["x_break"]
        if diff > 0: res += r["slope"] * diff
    return res

# --- Content-Addressed M1 LoRA Morphism (Pillars B1, B7) ---
class LoRAMorphism:
    """Content-addressed rank-r delta adapter: W' = W + (alpha / r) A B^T."""
    def __init__(self, alpha: float, r: int, A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]):
        self.alpha, self.r = alpha, r
        self.A = [list(row) for row in A] # M x r
        self.B = [list(row) for row in B] # N x r

    def contract(self, x: Sequence[float]) -> List[float]:
        # (A (B^T x)) * (alpha / r)
        scale = self.alpha / self.r
        # B^T x
        r_dim = self.r
        z = [sum(self.B[j][k] * x[j] for j in range(len(x))) for k in range(r_dim)]
        return [sum(self.A[i][k] * z[k] for k in range(r_dim)) * scale for i in range(len(self.A))]
