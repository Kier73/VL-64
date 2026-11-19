"""Optional genuine pretrained *range-retention* and scalar-reference experiment.

Runs already independently checked Qwen trained-head / DeepSeek selected-slice
subgraphs twice through one persisted, bounded observer-range cache. The second
pass is deliberately the same input; this proves pretrained parameter reuse,
NOT causal autoregressive continuation or a complete large-model layer.
Offline tests cover true causal Qwen continuation using the same cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .qualified_range_cache import QualifiedRangeCache
from .scaled_model_fabric import MAX_INDEX_BYTES, QualifiedShardNavigator
from .scaled_model_lab import PINNED, _url, _fetch_metadata, pinned_hf_range_host
from .qualified_subgraphs import (
    probe_qwen_trained_delta_head, probe_deepseek_trained_router,
    probe_deepseek_selected_fp4_expert_slices,
)

PROVIDER = 'pinned-huggingface-https-range'


def run(model: str, *, cache_dir, budget_bytes=64 * 1024 * 1024):
    """Return an attested byte-reuse report, or propagate failure to CLI."""
    if model not in ('qwen', 'deepseek'):
        raise ValueError('Only pinned Qwen and DeepSeek models are qualified')
    model_name, revision = PINNED[model]
    index = _fetch_metadata(_url(model_name, revision, 'model.safetensors.index.json'),
                            MAX_INDEX_BYTES)
    cache = QualifiedRangeCache(cache_dir, pinned_hf_range_host(model_name, revision),
                                approved_hosts={PROVIDER}, budget_bytes=budget_bytes)
    nav = QualifiedShardNavigator(model=model_name, revision=revision, index_raw=index,
                                  range_host=cache, approved_hosts={PROVIDER})
    def observe():
        if model == 'qwen':
            receipt = probe_qwen_trained_delta_head(nav)
            if receipt['status'] != 'REAL_QWEN_GATED_DELTA_TRAINED_HEAD_VERIFIED':
                raise ValueError('Pretrained Qwen head reference did not verify')
            return {'scope': 'one real trained head, not a complete GDN layer',
                    'reference_error': receipt['max_abs_independent_error'],
                    'numerical_output': receipt['output_row0_head_contribution'],
                    'head': receipt['value_head']}
        router = probe_deepseek_trained_router(nav)
        if router['status'] != 'REAL_DEEPSEEK_TRAINED_384_ROUTER_VERIFIED':
            raise ValueError('Pretrained DeepSeek router did not verify')
        slices = probe_deepseek_selected_fp4_expert_slices(nav, router)
        if slices['status'] != 'REAL_DEEPSEEK_SELECTED_FP4_SWIGLU_SLICES_VERIFIED':
            raise ValueError('DeepSeek selected expert FP4 slices did not verify')
        return {'scope': 'router and six selected expert row fragments, not full MoE',
                'reference_error': max(router['max_abs_independent_error'],
                                       slices['max_abs_independent_error']),
                'selected_experts': router['selected_experts'],
                'selected_slice_receipt': slices['status']}
    first = observe()
    at_first = cache.stats()
    first_pass_tensor_bytes = nav.weight_bytes
    second = observe()
    final = cache.stats()
    if first != second:
        raise ValueError('Retained weights produced a different numerical reference')
    if final['qualified_remote_ranges'] != at_first['qualified_remote_ranges']:
        raise ValueError('Repeated pretrained subgraph required additional remote ranges')
    if final['retained_cache_hits'] <= at_first['retained_cache_hits']:
        raise ValueError('No admitted range was reused')
    if final['qualified_remote_bytes'] < first_pass_tensor_bytes:
        raise ValueError('Network transfer accounting differs from tensor payload evidence')
    if first['reference_error'] > 2e-5:
        raise ValueError('Pretrained independent scalar reference disagreement')
    head = cache.snapshot()
    return {'schema': 'vl64.real-pretrained-range-retention.v1',
            'status': 'REAL_PINNED_WEIGHT_REUSE_AND_REFERENCE_VERIFIED',
            'model': model_name, 'revision': revision,
            'index_ground_digest': nav.index['ground_digest'],
            'subgraph': first, 'repeat_count': 2,
            'remote_ranges_after_first': at_first['qualified_remote_ranges'],
            'remote_ranges_after_second': final['qualified_remote_ranges'],
            'remote_transfer_bytes_total': final['qualified_remote_bytes'],
            'first_pass_tensor_bytes': first_pass_tensor_bytes,
            'remote_header_bytes': final['qualified_remote_bytes']-first_pass_tensor_bytes,
            'reused_range_requests': (final['retained_cache_hits']-
                                      at_first['retained_cache_hits']),
            'original_witness_cache_head': head,
            'range_transcript_digest': nav.chain,
            'scope': 'same-input weight reuse and scalar reference; '
                     'not independent real full-layer inference or causal text generation'}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=('qwen', 'deepseek'))
    p.add_argument('--out', default='artifacts/retained-weights.json')
    p.add_argument('--cache-dir', default='artifacts/qualified-weight-cache')
    p.add_argument('--budget-bytes', type=int, default=64 * 1024 * 1024)
    args = p.parse_args(argv)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args.model, cache_dir=args.cache_dir,
                     budget_bytes=args.budget_bytes)
    except (ValueError, OSError, TimeoutError, OverflowError) as exc:
        report = {'status': 'UNRESOLVED', 'model': PINNED[args.model][0],
                  'error_type': type(exc).__name__,
                  'reason': str(exc)[:200],
                  'scope': 'no real pretrained range reuse claim without actual bytes'}
    path.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    print('VL64_RETAINED_PRETRAINED '+json.dumps(report, sort_keys=True))
    return 0 if report['status'] == 'REAL_PINNED_WEIGHT_REUSE_AND_REFERENCE_VERIFIED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
