"""Tests for observed byte-to-Ground translation and empirical navigation."""
from pathlib import Path
import math

import pytest
import vld
from vld import ObservedGround


def cycle_samples():
    # Counter is predictable, second byte varies between observations.
    training = [
        (bytes((i, 16 * t + i)), bytes(((i + 1) % 16, 80 + 16 * t + i)))
        for t in range(4) for i in range(16)
    ]
    validation = [
        (bytes((i, 160 + i)), bytes(((i + 1) % 16, 192 + i)))
        for i in range(16)
    ]
    return training, validation


def test_lift_and_ground_identity():
    ground = ObservedGround({"full": None, "register": (0,), "word": (0, 1)})
    raw = bytes((7, 255, 0, 10))
    observation = ground.lift(raw)
    assert observation["byte_count"] == 4
    assert observation["states"] == {
        "full": "07ff000a", "register": "07", "word": "07ff"
    }
    assert observation["byte_digest"] != ground.lift(raw[::-1])["byte_digest"]
    assert ground.fingerprint() == ObservedGround(
        {"full": None, "register": [0], "word": [0, 1]}
    ).fingerprint()


def test_projection_validation():
    with pytest.raises(ValueError):
        ObservedGround({})
    with pytest.raises(ValueError):
        ObservedGround({"bad": (0, 0)})
    with pytest.raises(ValueError):
        ObservedGround({"bad": (-1,)})
    ground = ObservedGround({"register": (4,)})
    with pytest.raises(ValueError):
        ground.lift(b"abc")
    with pytest.raises(TypeError):
        ground.lift("abc")


def test_model_selection_and_empirical_information():
    training, validation = cycle_samples()
    ground = ObservedGround({"full": None, "counter": (0,), "noise": (1,)})
    ground.fit(training)
    profile = ground.profile("counter")
    assert profile["pairs"] == 64
    assert profile["source_states"] == 16
    assert profile["target_states"] == 16
    assert profile["single_successor_sources"] == 16
    assert profile["multiple_successor_sources"] == 0
    assert profile["conditional_entropy_bits"] == 0.0
    assert profile["predictive_information_bits"] == 4.0
    ranked = ground.evaluate(validation)
    assert ranked[0]["view"] == "counter"
    assert ranked[0]["accuracy"] == 1.0
    assert ranked[0]["coverage"] == 1.0
    assert ranked[0]["score"] == 4.0
    assert next(v for v in ranked if v["view"] == "full")["coverage"] == 0.0
    assert ground.predict(b"\x04\xff", "counter")["outcomes"] == {"05": 4}


def test_empirical_uncertainty_and_entropy_change():
    divergent = ObservedGround({"byte": (0,)})
    divergent.fit([(b"\x00", b"\x01"), (b"\x00", b"\x02")])
    prediction = divergent.predict(b"\x00", "byte")
    assert prediction["status"] == "MULTIPLE_OBSERVED_SUCCESSORS"
    assert prediction["conditional_entropy_bits"] == 1.0
    assert divergent.predict(b"\x09", "byte")["status"] == "UNOBSERVED"
    assert divergent.profile("byte")["entropy_change_bits"] == 1.0
    with pytest.raises(ValueError, match="ambiguous"):
        divergent.navigate(b"\x00", 1, "byte")
    with pytest.raises(ValueError, match="missing"):
        divergent.navigate(b"\x09", 1, "byte")
    convergent = ObservedGround().fit([(b"\x00", b"\x03"), (b"\x01", b"\x03")])
    assert convergent.profile("bytes")["entropy_change_bits"] == -1.0


def test_trillion_step_jump_from_observed_cycle():
    train, _ = cycle_samples()
    ground = ObservedGround({"counter": (0,)}).fit(train)
    n = 10**12 + 5
    result = ground.navigate(bytes((3, 200)), n, "counter")
    assert result["status"] == "OBSERVED_MODEL_NAVIGATION"
    assert result["terminal_state"] == f"{(3 + n) % 16:02x}"
    assert result["cycle_length"] == 16
    assert result["evaluated_steps"] <= 16
    assert result["evaluated_steps"] + result["skipped_steps"] == n
    assert result["ground_digest"] == ground.fingerprint()
    assert ground.navigate(bytes((3, 200)), 0, "counter")["terminal_state"] == "03"
    with pytest.raises(ValueError):
        ground.navigate(bytes((3, 200)), -1, "counter")


def test_fingerprint_tracks_observed_evidence():
    g = ObservedGround()
    initial = g.fingerprint()
    g.fit([(b"a", b"b")])
    current = g.fingerprint()
    assert initial != current
    g.fit([(b"a", b"b")])
    assert g.fingerprint() != current
    assert g.fingerprint() == ObservedGround().fit(
        [(b"a", b"b"), (b"a", b"b")]
    ).fingerprint()


def test_automatic_view_discovery():
    training, validation = cycle_samples()
    model, ranking = ObservedGround.discover(
        training, validation, max_positions=2, max_fields=2
    )
    assert ranking[0]["view"] == "byte_0"
    assert ranking[0]["score"] == 4.0
    assert next(item for item in ranking if item["view"] == "bytes")["coverage"] == 0
    nav = model.navigate(bytes((5, 250)), 10**12, "byte_0")
    assert nav["terminal_state"] == f"{(5 + 10**12) % 16:02x}"
    with pytest.raises(ValueError):
        ObservedGround.discover([], validation)
    with pytest.raises(ValueError):
        ObservedGround.discover(training, validation, max_positions=0)


def test_byte_budget_including_translation():
    src = Path(vld.__file__).resolve().parent
    package_bytes = sum(f.stat().st_size for f in src.glob("*.py"))
    assert package_bytes <= 65_536
    print("VL64_TRANSLATION_SOURCE_BYTES", package_bytes)
