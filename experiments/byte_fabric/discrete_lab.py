"""Use the public repository as a byte substrate for a controlled non-affine machine.

The state dynamics are independent synthetic reference data. The byte template
comes from VL-64's checked-out README. We do not claim the README itself is
a remotely changing state machine.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

from vld import ObserverIntention
from .ground_families import GroundFamilies
from .quotient_ground import compile_observable_quotient


def sequence(count=192,template=None):
    if type(count) is not int or count<100:
        raise ValueError("Require a meaningful observation horizon")
    if template is None:
        # Repository bytes provide nontrivial real-world constant context.
        template=Path("README.md").read_bytes()[:1024]
    if len(template)<500:
        raise ValueError("Not enough byte template material")
    template=template[:1024]
    offset=427
    variable_width=2
    template=template[:offset]+bytes(variable_width)+template[offset+variable_width:]
    tokens=[hashlib.sha256(
        ("vl64-observer-nonlinear-cycle/"+str(i)).encode()
        ).digest()[:variable_width] for i in range(20)]
    if len(set(tokens))!=20:
        raise RuntimeError("Reference states accidentally collided")
    prefix,cycle=3,17
    series=[tokens[i if i<prefix else prefix+(i-prefix)%cycle]
            for i in range(count)]
    frames=[template[:offset]+state+template[offset+variable_width:]
            for state in series]
    successor={series[i]:series[i+1] for i in range(prefix+cycle)}
    return frames,successor,prefix,cycle


def experiment():
    frames,successor,prefix,cycle=sequence()
    begin=time.perf_counter_ns()
    selected=GroundFamilies().discover(frames[:58],frames[58:110])
    discovery_ms=(time.perf_counter_ns()-begin)/1e6
    ground=selected["ground"]
    machine=GroundFamilies().open(ground)
    unseen=[110,133,159,191]
    for index in unseen:
        result=machine.realize(index,"bytes.snapshot",include_data=True)
        if base64.b64decode(result["result"]["data_b64"])!=frames[index]:
            raise RuntimeError("Previously unseen reference state does not reconstruct")
    horizon=10**12
    predicted=machine.realize(horizon,"digest.sha256")
    if not machine.check_against_core(frames[:58],horizon):
        raise RuntimeError("Compact ObservedGround disagrees with compiled finite model")
    sample_horizon=300000
    start=time.perf_counter_ns()
    state=frames[0][427:429]
    for _ in range(sample_horizon):
        state=successor[state]
    reference_ms=(time.perf_counter_ns()-start)/1e6
    start=time.perf_counter_ns()
    jumped=machine.realize(sample_horizon,"state.token")
    jump_ms=(time.perf_counter_ns()-start)/1e6
    if jumped["result"]["hex"]!=state.hex():
        raise RuntimeError("Navigated transition conflicts with reference stepping")
    good=machine.test_new_observations(
        [(index,frames[index]) for index in unseen])
    altered=bytearray(frames[159])
    altered[300] ^= 1
    bad=machine.test_new_observations([(159,bytes(altered))])
    if good["status"]!="CONSISTENT_ON_NEW_OBSERVATIONS" or bad["status"]!="CONTRADICTED":
        raise RuntimeError("Did not preserve ground contradiction evidence")
    return {
        "ground":ground,
        "report":{
            "schema":"vl64.nonaffine-discrete-ground-measurement.v1",
            "status":"HELDOUT_AND_NEW_OBSERVATIONS_RECONSTRUCTED",
            "reference_source":"VL-64 README static template plus controlled synthetic state changes",
            "selected_family":selected["selection"]["chosen_family"],
            "family_candidates":selected["selection"]["candidates"],
            "training_frames":58,"heldout_frames":52,
            "new_unseen_indices":unseen,
            "frame_bytes":len(frames[0]),
            "raw_observed_bytes":selected["measurement"]["raw_observed_bytes"],
            "ground_description_bytes":selected["measurement"]["ground_description_bytes"],
            "compression_ratio":selected["measurement"]["compression_ratio"],
            "discovered_state_span":{
                "offset":ground["variable_offset"],"width":ground["variable_width"]},
            "prefix_length":prefix,"cycle_length":cycle,
            "trillion_horizon":horizon,
            "trillion_result_digest":predicted["result"]["value"],
            "horizon_relation_count":len(predicted["stage_receipts"]),
            "reference_stepping_horizon":sample_horizon,
            "reference_stepping_ms":round(reference_ms,4),
            "discrete_navigation_ms":round(jump_ms,4),
            "speedup_one_run":round(reference_ms/max(jump_ms,0.000001),2),
            "ground_discovery_ms":round(discovery_ms,4),
            "matched_challenge":good["status"],
            "contradictory_challenge":bad["status"],
            "core_crosscheck":True,
            "future_external_behavior_proven":False,
            "core_modified":False,
        },
    }



def quotient_experiment():
    """Independent observer asks for a coarser future-stable output state."""
    template=(b"Internet byte observation/recognition/"*20)[:512]
    offset=171
    template=template[:offset]+b"\0"+template[offset+1:]
    tokens=[bytes([50+i]) for i in range(12)]
    frames=[
        template[:offset]+tokens[i%12]+template[offset+1:]
        for i in range(96)
    ]
    source=GroundFamilies().discover(frames[:38],frames[38:70])
    driver=GroundFamilies().open(source["ground"])
    if source["selection"]["chosen_family"]!="closed-finite-cycle":
        raise RuntimeError("Expected non-affine finite reference")
    labels={token.hex():("A" if i%3 in (0,1) else "B")
            for i,token in enumerate(driver.tokens)}
    quotient=compile_observable_quotient(driver,labels)
    model=GroundFamilies().open(quotient["ground"])
    distant=model.realize(10**12,"state.observable")
    expected=labels[tokens[(10**12)%12].hex()]
    if distant["result"]["value"]!=expected:
        raise RuntimeError("Quotient-model navigation disagrees with reference")
    for i in range(70,96):
        if model.realize(i,"state.observable")["result"]["value"] != labels[
                tokens[i%12].hex()]:
            raise RuntimeError("Unseen observable continuation disagrees")
    good=model.test_new_observations([(72,labels[tokens[72%12].hex()])])
    bad=model.test_new_observations([(72,"CONTRADICTION")])
    if good["status"]!="CONSISTENT_ON_NEW_OBSERVATIONS" or bad[
            "status"]!="CONTRADICTED":
        raise RuntimeError("Observable quotient challenge policy failed")
    return {
        "ground":quotient["ground"],
        "report":{
            "schema":"vl64.observation-quotient-experiment.v1",
            "status":"OBSERVABLE_FUTURES_PRESERVED_IN_DECLARED_MODEL",
            "source_states":quotient["measurement"]["source_states"],
            "quotient_states":quotient["measurement"]["quotient_states"],
            "erasure_entropy_bits":quotient["measurement"]["erasure_entropy_bits"],
            "original_observed_snapshot_bytes":70*len(frames[0]),
            "quotient_description_bytes":quotient["measurement"]["description_bytes"],
            "fresh_observable_continuations_checked":26,
            "horizon":10**12,
            "horizon_observed_value":distant["result"]["value"],
            "exact_source_bytes_reconstructible":False,
            "matched_challenge":good["status"],
            "contradictory_challenge":bad["status"],
            "meaning":"observer-output-equivalence-not-full-byte-reconstruction",
        }
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="artifacts")
    args=parser.parse_args()
    output=Path(args.out)
    output.mkdir(parents=True,exist_ok=True)
    result=experiment()
    q=quotient_experiment()
    (output/"learned-observable-quotient-ground.json").write_text(
        json.dumps(q["ground"],sort_keys=True,indent=2)+"\n")
    (output/"observable-quotient-measurements.json").write_text(
        json.dumps(q["report"],sort_keys=True,indent=2)+"\n")
    quotient_request=ObserverIntention(
        "ground.navigate",
        "workspace:"+(output/"learned-observable-quotient-ground.json").as_posix(),
        output="json",
        options={"goal":{"kind":"state.observable","steps":10**12,
                         "max_work":2,"max_hops":2}})
    (output/"observable-quotient-intention.json").write_text(
        json.dumps(quotient_request.to_dict(),sort_keys=True,indent=2)+"\n")
    ground_path=output/"learned-discrete-ground.json"
    ground_path.write_text(json.dumps(result["ground"],sort_keys=True,indent=2)+"\n")
    request=ObserverIntention(
        "ground.navigate",
        "workspace:"+ground_path.as_posix(),
        options={"goal":{"kind":"digest.sha256","steps":10**12,
                         "max_work":4,"max_hops":4}},
        output="json")
    (output/"nonaffine-observer-intention.json").write_text(
        json.dumps(request.to_dict(),sort_keys=True,indent=2)+"\n")
    (output/"nonaffine-ground-measurements.json").write_text(
        json.dumps(result["report"],sort_keys=True,indent=2)+"\n")
    print("VL64_NONAFFINE_REPORT "+json.dumps(result["report"],sort_keys=True))
    print("VL64_NONAFFINE_GROUND "+json.dumps(result["ground"],sort_keys=True,
                                           separators=(",",":")))
    print("VL64_QUOTIENT_REPORT "+json.dumps(q["report"],sort_keys=True))
    print("VL64_QUOTIENT_GROUND "+json.dumps(q["ground"],sort_keys=True,
                                         separators=(",",":")))


if __name__=="__main__":
    main()
