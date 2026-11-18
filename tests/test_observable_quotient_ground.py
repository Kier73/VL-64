"""Observation congruence: preserve future observable behavior, not raw states."""
import hashlib
import json

import pytest
from vld import ObserverIntention
from experiments.byte_fabric.discrete_ground import (
    discover_discrete_ground,DiscreteGroundNavigator,
)
from experiments.byte_fabric.quotient_ground import (
    compile_observable_quotient,ObservableQuotientNavigator,
)
from experiments.byte_fabric.ground_families import GroundFamilies
from experiments.observer_intent.arbitrate import arbitrate


def source_frames(n=90):
    tokens=[bytes([40+i]) for i in range(12)]
    template=b"OBSERVED-MACHINE-" + (b"z"*350)
    offset=29
    template=template[:offset]+b"\0"+template[offset+1:]
    frames=[template[:offset]+tokens[i%12]+template[offset+1:]
            for i in range(n)]
    return frames,tokens


def models():
    frames, tokens=source_frames()
    ground=discover_discrete_ground(frames[:38],frames[38:68])["ground"]
    driver=DiscreteGroundNavigator(ground)
    labels={t.hex():("A" if i%3 in (0,1) else "B")
            for i,t in enumerate(driver.tokens)}
    quotient=compile_observable_quotient(driver,labels)
    return frames,driver,labels,quotient


def test_observational_congruence_is_stronger_than_current_value_equality():
    frames,driver,labels,q=models()
    assert q["measurement"]["source_states"]==12
    # Current-output grouping would have only 2 labels; future behavior
    # separates the two 'A' phases and gives 3 stable state classes.
    assert len(set(labels.values()))==2
    assert q["measurement"]["quotient_states"]==3
    assert q["measurement"]["states_merged"]==9
    assert q["measurement"]["erasure_entropy_bits"]==2.0
    assert q["measurement"]["byte_reconstruction_qualified"] is False
    assert q["ground"]["reconstruction_claim"]["exact_source_bytes_recoverable_from_quotient_alone"] is False
    driver_q=ObservableQuotientNavigator(q["ground"])
    for i in list(range(40))+[10**12,10**12+1,10**12+2]:
        observed=driver_q.realize(i,"state.observable")
        assert observed["status"]=="OBSERVED_GROUND_NAVIGATION"
        assert observed["result"]["value"]==labels[driver._token_at(i).hex()]
    assert driver_q.realize(3,"bytes.snapshot")["status"]=="UNRESOLVED"


def test_quotient_cannot_merge_two_equal_outputs_with_unequal_futures():
    _,driver,labels,q=models()
    classes=q["ground"]["classes"]
    assert len(classes)==3
    assert sorted(c["source_states"] for c in classes)==[4,4,4]
    successors={c["next"] for c in classes}
    assert len(successors)==3
    assert set(c["label"] for c in classes)=={"A","B"}


def test_one_label_collapses_everything_without_claiming_bytes():
    _,driver,_,_=models()
    labels={token.hex():"same" for token in driver.tokens}
    quotient=compile_observable_quotient(driver,labels)
    assert quotient["measurement"]["quotient_states"]==1
    assert quotient["measurement"]["states_merged"]==11
    assert quotient["measurement"]["byte_reconstruction_qualified"] is False
    q=ObservableQuotientNavigator(quotient["ground"])
    assert q.realize(10**12,"state.observable")["result"]["value"]=="same"


def test_quotient_challenges_are_observable_only_and_detect_contradiction():
    _,_,_,q=models()
    driver=ObservableQuotientNavigator(q["ground"])
    expected=driver.realize(10**12,"state.observable")["result"]["value"]
    assert driver.test_new_observations(
        [(10**12,expected)])["status"]=="CONSISTENT_ON_NEW_OBSERVATIONS"
    assert driver.test_new_observations(
        [(10**12,"different")])["status"]=="CONTRADICTED"


def test_quotient_can_be_reopened_as_portable_ground_and_observer_intention(tmp_path):
    _,driver,_,q=models()
    assert isinstance(GroundFamilies().open(q["ground"]),ObservableQuotientNavigator)
    (tmp_path/"quotient.json").write_text(json.dumps(q["ground"]))
    req=ObserverIntention(
        "ground.navigate","workspace:quotient.json",
        options={"goal":{"steps":10**12,"kind":"state.observable",
                         "max_work":2,"max_hops":2}})
    result=arbitrate(req,byte_root=tmp_path)
    assert result["status"]=="OBSERVED_GROUND_NAVIGATION"
    assert result["navigation"]["result"]["value"] in ("A","B")


def test_invalid_labels_or_broken_quotient_rejected():
    _,driver,labels,q=models()
    with pytest.raises(ValueError):
        compile_observable_quotient(driver,{})
    with pytest.raises(ValueError):
        compile_observable_quotient(driver,{**labels,"unknown":"A"})
    forged=dict(q["ground"])
    forged["classes"]=forged["classes"][:-1]
    with pytest.raises(ValueError):
        ObservableQuotientNavigator(forged)
    forged=dict(q["ground"])
    forged["classes"]=[dict(c) for c in forged["classes"]]
    forged["classes"][0]["next"]=999
    from vld import domain_digest
    forged["ground_digest"]=domain_digest("vl64.observable-quotient-ground.v1",
        {k:v for k,v in forged.items() if k!="ground_digest"})
    with pytest.raises(ValueError):
        ObservableQuotientNavigator(forged)
