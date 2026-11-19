"""CPU-only VL64 intent fleet; real Qwen/DeepSeek arithmetic stays remote."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .remote_large_fleet import evaluate_fleet, fixture_transport


def main(argv=None):
    parser=argparse.ArgumentParser(description='External large-model VL64 controller')
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--fixture',action='store_true',help='No trained model inference; protocol tests only')
    modes.add_argument('--live',action='store_true',help='Use authenticated full-scale remote providers')
    parser.add_argument('--max-tokens',type=int,default=160)
    parser.add_argument('--instance',choices=('qwen-27b','deepseek-v41-flash'),
                        help='Run one independent large-model VL interpreter instance')
    parser.add_argument('--out',default='artifacts/remote-large-fleet.json')
    args=parser.parse_args(argv)
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    try:
        report=evaluate_fleet(live=args.live,
                transport_factory=None if args.live else fixture_transport,
                max_tokens=args.max_tokens,
                routes=(args.instance,) if args.instance else None)
    except (ValueError,RuntimeError,OSError,TimeoutError,TypeError) as exc:
        report={'schema':'vl64.large-instruction-fleet-benchmark.v1',
                'status':'UNRESOLVED', 'exception':type(exc).__name__,
                'mode':'live_external_large_models' if args.live else 'control_fixture',
                'claim':'no_model_inference_without_valid_remote_host_evidence'}
    target.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('VL64_LARGE_MODEL_FLEET '+json.dumps({
        'status':report['status'],'model_routes':list(report.get('instances',[])[i]['route']
                        for i in range(len(report.get('instances',[]))))},sort_keys=True))
    return 1 if report['status']=='UNRESOLVED' else 0

if __name__ == '__main__':
    raise SystemExit(main())
