"""Optional real BF16 model-orbit checkpoint/rehome proof on pinned SmolLM2.

The HTTP range server supplies original pretrained BF16 tensor bytes. The
execution happens in one local Python process and the 'observer IDs' are local
routing assignments, not two distinct physical machines. A separate fixture
suite proves actual process restart, strict external head pins and no prefix
replay. No claimed full-model logits or remote math execution.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

from .execution_dag import compile_decoder_dag, huggingface_range_host
from .pretrained_orbit import ContentAddressedPages, RecoverableModelOrbit, save_checkpoint, load_checkpoint
from .real_chain_lab import MANIFEST, independent_two_layer_reference

TOKENS = (42, 75)


def run(*, directory=None, tokens=TOKENS):
    if len(tokens) != 2:
        raise ValueError("This experiment requires exactly two selected pretrained token IDs")
    observed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    plan = compile_decoder_dag(observed)
    remote = huggingface_range_host(plan)
    captured = defaultdict(list)
    observed_bytes = [0, 0]
    observed_ranges = [0, 0]

    def source(stage):
        def fetch(request):
            reply = remote(request)
            captured[request["tensor"]].append((request["byte_begin"],reply["payload"]))
            observed_bytes[stage] += len(reply["payload"])
            observed_ranges[stage] += 1
            return dict(reply, observer_host=("checkpoint-source-a" if stage == 0
                                               else "checkpoint-source-b"))
        return fetch

    def execute_with_store(root):
        pages = ContentAddressedPages(root/"pages")
        a = source(0)
        first = RecoverableModelOrbit(plan,{-1:a,0:a,1:a},
            approved_hosts={-1:{"checkpoint-source-a"},0:{"checkpoint-source-a"},
                            1:{"checkpoint-source-a"}},
            compute_actors={0:"local-interpretation-a"},
            approved_producers={"local-interpretation-a"},
            store=pages,owner="observer-assignment-a",epoch=0,last_layer=1)
        first_output = first.advance(tokens[0],expected_position=0)
        checkpoint = first.checkpoint()
        pinned_head = save_checkpoint(root/"causal-checkpoint.json",checkpoint)
        b = source(1)
        recovered, verified_hidden = RecoverableModelOrbit.restore(
            plan, load_checkpoint(root/"causal-checkpoint.json"),
            expected_head=pinned_head,new_owner="observer-assignment-b",new_epoch=1,
            range_hosts={-1:b,0:b,1:b},
            approved_hosts={-1:{"checkpoint-source-b"},0:{"checkpoint-source-b"},
                            1:{"checkpoint-source-b"}},
            compute_actors={0:"local-interpretation-b"},
            approved_producers={"local-interpretation-b"},store=pages)
        if verified_hidden != first_output["state"]:
            raise ValueError("Restored pretrained hidden vector mismatches committed position")
        second_output = recovered.advance(tokens[1],expected_position=1)
        reference = independent_two_layer_reference(plan,captured,tokens,last_layer=1)
        residual = max(abs(x-y) for x,y in zip(second_output["state"],reference))
        tolerance = 2e-5
        if not math.isfinite(residual) or residual>tolerance:
            raise ValueError("Restored pretrained orbit diverges from exact-byte scalar reference")
        new_checkpoint=recovered.checkpoint()
        if (new_checkpoint["previous_checkpoint"]!=pinned_head or
                new_checkpoint["epoch"]!=1 or new_checkpoint["tokens"]!=list(tokens)):
            raise ValueError("Restored model orbit lost causal checkpoint lineage")
        if checkpoint["static_weight_identities"]!=new_checkpoint["static_weight_identities"]:
            raise ValueError("Rehomed pinned model changed pretrained layer weights")
        report={
            "schema":"vl64.real-pretrained-recovered-orbit.v2",
            "status":"REAL_BF16_TWO_LAYER_RESTART_CONTINUITY_VERIFIED",
            "model":plan["model"],"revision":plan["revision"],
            "dag_digest":plan["dag_digest"],"tokens":list(tokens),
            "layers_executed":[0,1],"checkpoint_head":pinned_head,
            "next_checkpoint_head":new_checkpoint["checkpoint_digest"],
            "old_owner":checkpoint["owner"],"new_owner":new_checkpoint["owner"],
            "old_epoch":checkpoint["epoch"],"new_epoch":new_checkpoint["epoch"],
            "static_weight_identities":new_checkpoint["static_weight_identities"],
            "weight_range_witness_chains_before":checkpoint["weight_witness_chains"],
            "weight_range_witness_chains_after":new_checkpoint["weight_witness_chains"],
            "exact_range_reads_by_assignment":observed_ranges,
            "acquired_BF16_bytes_by_assignment":observed_bytes,
            "independent_scalar_reference_max_abs_error":round(residual,12),
            "tolerance_abs":tolerance,
            "last_hidden_first_8":[round(x,8) for x in second_output["state"][:8]],
            "evidence_scope":"external_expected_head_plus_hash_verified_KV_pages_and_pinned_HTTP_ranges",
            "compute_scope":"one_python_interpreter_two_logical_observer_assignments_not_remote_processor_handoff",
            "model_scope":"two_pretrained_layers_not_complete_model_logits_or_text_generation",
        }
        (root/"orbit-report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        return report
    if directory is None:
        with tempfile.TemporaryDirectory(prefix="vl64-real-orbit-") as temp:
            return execute_with_store(Path(temp))
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    return execute_with_store(root)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="artifacts/pretrained-orbit-report.json")
    parser.add_argument("--state-dir",default="artifacts/pretrained-orbit-pages")
    args=parser.parse_args()
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    try:
        report=run(directory=Path(args.state_dir))
    except (ValueError,OSError,TimeoutError) as exc:
        report={"status":"UNRESOLVED","reason":type(exc).__name__,
                "scope":"no_real_pretrained_recovery_claim_without_exact_observed_weights"}
        out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        print("VL64_RECOVERABLE_PRETRAINED_ORBIT "+json.dumps(report,sort_keys=True))
        raise SystemExit(1) from exc
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("VL64_RECOVERABLE_PRETRAINED_ORBIT "+json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
