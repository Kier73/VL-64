"""μVLD-64: Consolidated 64 KiB Universal Virtual Layer Dynamics Runtime.

This package fits the complete mathematical, cryptographic, substrate, and neural capabilities
of Virtual Layer Dynamics into an exact <= 64 KiB code budget.
"""
from .micro_kernel import (
    canonical_bytes, domain_digest,
    ed25519_keypair_from_seed, ed25519_sign, ed25519_verify,
    k3_and, k3_or, k3_not, classify_interval_guard,
    K3_FALSE, K3_INDETERMINATE, K3_TRUE,
    myhill_nerode_quotient, monoid_power, affine_compose, affine_navigate,
    certify_modular_progression, ExactFieldMatrix, solve_sheaf_cohomology,
    syndrome_decode_sparse_fault, HashChainedJournal, ScopedVectorClock, EGraph
)

from .micro_operators import (
    StructuralLinearOperator,
    dequant_e4m3_byte, dequant_ue8m0_byte, dequant_mxfp4_nibble, dequant_block_fp8,
    sinkhorn_birkhoff_substochastic, rms_norm, FlashStreamState,
    linear_cka_consensus, PrecisionCertificate,
    compile_polyhedral_hinge_1d, eval_polyhedral_hinge_1d, LoRAMorphism
)

from .micro_substrates import (
    ByteLens, LENS_RAW, LENS_UTF8, LENS_U32LE,
    PageSpace, GitWireV2Parser, HeadlessOS, SafetensorsGlance,
    topk_moe_routing, StreamPathMachine
)

from .micro_control import (
    NavigationCostVector, IntentExpr, IntentionSpec,
    EffectLedger, IntentionCompiler, QuorumAuthority,
    meta_arbitrator_utility, AutopoieticDiscoveryLoop,
    verify_bundle_standalone, run_self_verification, GroundRegistry
)

__version__ = "0.7.0"
__all__ = [
    "canonical_bytes", "domain_digest",
    "ed25519_keypair_from_seed", "ed25519_sign", "ed25519_verify",
    "k3_and", "k3_or", "k3_not", "classify_interval_guard",
    "myhill_nerode_quotient", "monoid_power", "affine_navigate",
    "certify_modular_progression", "ExactFieldMatrix", "solve_sheaf_cohomology",
    "syndrome_decode_sparse_fault", "HashChainedJournal", "ScopedVectorClock", "EGraph",
    "StructuralLinearOperator", "dequant_block_fp8", "sinkhorn_birkhoff_substochastic",
    "rms_norm", "FlashStreamState", "linear_cka_consensus", "PrecisionCertificate",
    "compile_polyhedral_hinge_1d", "eval_polyhedral_hinge_1d", "LoRAMorphism",
    "ByteLens", "PageSpace", "GitWireV2Parser", "HeadlessOS", "SafetensorsGlance",
    "topk_moe_routing", "StreamPathMachine",
    "NavigationCostVector", "IntentExpr", "IntentionSpec", "EffectLedger",
    "IntentionCompiler", "QuorumAuthority", "meta_arbitrator_utility",
    "AutopoieticDiscoveryLoop", "verify_bundle_standalone", "run_self_verification", "GroundRegistry"
]
