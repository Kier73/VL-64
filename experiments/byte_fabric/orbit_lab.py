"""A real 'use the system' controlled byte-state-compression experiment.

An independent state emitter supplies exact fixed-layout byte frames. VL-64
learns a compact hypothesis from training transitions, checks held-out frames,
then navigates and reconstructs states through a typed, multi-hop Ground.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

from vld import ObserverIntention, affine_navigate
from .orbit_ground import discover_ground, ComposedGroundNavigator


def generate_observed_frames(count=144):
    """Independent synthetic emitter, not the learned Ground interpreter."""
    offset=431
    width=4
    constant=(b"NET-OBSERVER-CAPACITY-TRANSIT-STATE|" * 32)[:1024]
    header=constant[:offset]
    trailer=constant[offset+width:]
    state=7
    emitted=[]
    for _ in range(count):
        payload=header+state.to_bytes(width,"little")+trailer
        emitted.append(payload)
        state=(1664525*state+1013904223)%(2**32)
    return emitted


def experiment():
    observed=generate_observed_frames(144)
    start=time.perf_counter_ns()
    # Discover which part of the frames changes and which interpretation
    # reproduces the held-out transitions; do not supply the offset or endian.
    fitted=discover_ground(observed[:32],observed[32:80])
    learned_ms=(time.perf_counter_ns()-start)/1000000
    ground=fitted["ground"]
    machine=ComposedGroundNavigator(ground)
    tested=[]
    # These frames are not supplied to training or validation at all.
    for i in (80,85,100,127,143):
        result=machine.realize(i,"bytes.snapshot",include_data=True)
        if base64.b64decode(result["result"]["data_b64"])!=observed[i]:
            raise RuntimeError("Unseen snapshot reconstruct failed")
        tested.append(i)
    # Compare the original iterative state-machine behavior at a horizon
    # short enough to compute directly, before navigating a distant horizon.
    short_horizon=200000
    initial=ground["initial_state"]
    start=time.perf_counter_ns()
    direct=initial
    for _ in range(short_horizon):
        direct=(1664525*direct+1013904223)%(2**32)
    direct_ms=(time.perf_counter_ns()-start)/1000000
    start=time.perf_counter_ns()
    shortcut=machine.realize(short_horizon,"state.scalar")
    shortcut_ms=(time.perf_counter_ns()-start)/1000000
    if direct!=shortcut["result"]["value"]:
        raise RuntimeError("Navigation did not reproduce the iterative machine")
    horizon=10**12
    start=time.perf_counter_ns()
    distant=machine.realize(horizon,"digest.sha256")
    navigate_ms=(time.perf_counter_ns()-start)/1000000
    if not machine.check_against_core(horizon):
        raise RuntimeError("Independent affine derivations disagree")
    ground_size=len(json.dumps(ground,sort_keys=True,separators=(",",":")).encode())
    raw_size=80*len(observed[0])
    return {
        "ground":ground,
        "report":{
            "schema":"vl64.compressive-ground-experiment.v1",
            "status":"VERIFIED_ON_OBSERVED_AND_UNSEEN_SAMPLES",
            "train_frames":32,"heldout_frames":48,
            "independent_unseen_frame_indices":tested,
            "frame_bytes":len(observed[0]),
            "raw_training_and_holdout_bytes":raw_size,
            "ground_description_bytes":ground_size,
            "compressed_ratio":round(raw_size/ground_size,3),
            "discovered_interpreter":fitted["discovery"]["selected"],
            "discovery_status":fitted["discovery"]["status"],
            "compression_scope":"exact_model_description_of_this_fixed-template_trace",
            "prediction_scope":"empirically_heldout_not_universal_future_proof",
            "horizon":horizon,
            "horizon_output_digest":distant["result"]["value"],
            "horizon_model_state":affine_navigate(
                ground["a"],ground["b"],ground["initial_state"],
                horizon,ground["modulus"]),
            "route":[s["name"] for s in distant["route"]["stages"]],
            "training_ms":round(learned_ms,4),
            "trillion_step_navigation_ms":round(navigate_ms,4),
            "direct_reference_horizon":short_horizon,
            "direct_reference_ms":round(direct_ms,4),
            "compressed_jump_ms":round(shortcut_ms,4),
            "short_horizon_speedup":round(direct_ms/max(shortcut_ms,0.000001),2),
            "basis_entries_cached":len(machine._powers),
            "core_source_modified":False,
        },
    }


def main():
    parser=argparse.ArgumentParser(description="Explore compressive executable byte Grounds")
    parser.add_argument("--out",default="artifacts")
    args=parser.parse_args()
    target=Path(args.out)
    target.mkdir(parents=True,exist_ok=True)
    results=experiment()
    ground_path=target/"learned-orbit-ground.json"
    ground_path.write_text(json.dumps(results["ground"],indent=2,sort_keys=True)+"\n")
    (target/"compressed-ground-report.json").write_text(
        json.dumps(results["report"],indent=2,sort_keys=True)+"\n")
    req=ObserverIntention("ground.navigate",
                          "workspace:"+ground_path.as_posix(),
                          output="json",
                          options={"goal":{
                              "kind":"digest.sha256",
                              "steps":10**12,
                              "max_work":5,
                              "max_hops":4,
                          }})
    (target/"trillion-step-intention.json").write_text(
        json.dumps(req.to_dict(),indent=2,sort_keys=True)+"\n")
    print("VL64_COMPRESSIVE_GROUND_REPORT "+
          json.dumps(results["report"],sort_keys=True))
    print("VL64_LEARNED_GROUND_JSON "+
          json.dumps(results["ground"],sort_keys=True,separators=(",",":")))


if __name__=="__main__":
    main()
