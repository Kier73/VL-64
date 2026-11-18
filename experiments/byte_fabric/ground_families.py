"""Select between independently validated algebraic and finite discrete Grounds.

The selection criterion is explicitly model-description bytes, after held-out
validation; it is not a universal semantic equivalence proof. The catalogue
can be extended with host-installed builders without changing the 64 KiB core.
"""
from __future__ import annotations

import json

from .discrete_ground import discover_discrete_ground, DiscreteGroundNavigator
from .orbit_ground import discover_ground, ComposedGroundNavigator


BUILDERS={
    "modular-affine":discover_ground,
    "closed-finite-cycle":discover_discrete_ground,
}
DRIVERS={
    "vl64.compressive-orbit-ground.v1":ComposedGroundNavigator,
    "vl64.discrete-orbit-ground.v1":DiscreteGroundNavigator,
}


class GroundFamilies:
    def __init__(self,builders=None,drivers=None):
        self.builders=dict(BUILDERS if builders is None else builders)
        self.drivers=dict(DRIVERS if drivers is None else drivers)

    def register_builder(self,name,builder):
        if (not isinstance(name,str) or not name or name in self.builders or
                not callable(builder)):
            raise ValueError("Invalid Ground builder registration")
        self.builders[name]=builder

    def register_driver(self,schema,driver):
        if (not isinstance(schema,str) or not schema or
                schema in self.drivers or not callable(driver)):
            raise ValueError("Invalid Ground interpreter registration")
        self.drivers[schema]=driver

    def open(self,ground):
        if not isinstance(ground,dict):
            raise ValueError("Expected typed Ground dictionary")
        schema=ground.get("schema")
        driver=self.drivers.get(schema)
        if driver is None:
            raise ValueError("No qualified Ground interpreter for schema")
        return driver(ground)

    def discover(self,training,heldout):
        training,heldout=list(training),list(heldout)
        candidates=[]
        observations=[]
        for name,builder in sorted(self.builders.items()):
            try:
                found=builder(training,heldout)
                ground=found["ground"]
                # The interpreter independently checks the descriptor's
                # integrity before the candidate becomes eligible.
                self.open(ground)
                size=len(json.dumps(ground,sort_keys=True,separators=(",",":"),
                                    allow_nan=False).encode())
                candidates.append((size,name,found))
                observations.append({
                    "candidate":name,"status":"HELDOUT_VALIDATED",
                    "description_bytes":size,
                    "schema":ground["schema"],
                    "ground_digest":ground["ground_digest"],
                })
            except (ValueError,TypeError,KeyError) as exc:
                observations.append({
                    "candidate":name,"status":"UNRESOLVED",
                    "reason":type(exc).__name__,
                    "detail":str(exc)[:160],
                })
        if not candidates:
            raise ValueError("No model family reproduced the held-out observations")
        size,model,selected=min(candidates,key=lambda c:(c[0],c[1]))
        return {
            "selection":{
                "status":"SELECTED_SHORTEST_HELDOUT_VALIDATED_DESCRIPTION",
                "chosen_family":model,
                "description_bytes":size,
                "candidates":observations,
                "scope":"model-selection-over-registered-families-only",
            },
            "ground":selected["ground"],
            "measurement":selected.get("measurement"),
            "discovery":selected.get("discovery"),
        }
