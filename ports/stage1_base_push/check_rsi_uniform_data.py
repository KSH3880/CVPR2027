"""Read-only prerequisite validation for the shareable PUSH RSI experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('weights/Stage1V2Fixed_00078000.pth'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    env=yaml.safe_load((root/'tokenhsi/data/cfg/multi_agent/stage1_push_rsi_uniform.yaml').read_text())['env']
    expected=set()
    def visit(value,parent):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('file','obj_file') and isinstance(item,str):expected.add(parent/item)
                else:visit(item,parent)
        elif isinstance(value,list):
            for item in value:visit(item,parent)
    for manifest in ('dataset_stage1_push_finetune.yaml','dataset_push_native_phase_xy.yaml'):
        path=root/'tokenhsi/data'/manifest
        visit(yaml.safe_load(path.read_text()),path.parent)
    asset=root/env['asset']['assetRoot']/env['asset']['assetFileName']
    expected.add(asset)
    if asset.exists():
        for mesh in ET.parse(asset).findall('./asset/mesh'):expected.add(asset.parent/mesh.get('file'))
    expected.add(args.source if args.source.is_absolute() else root/args.source)
    expected.add(root/env['openPushRSI']['bank_file'])
    missing=sorted(str(p) for p in expected if not p.is_file())
    if missing:
        print(f'Missing {len(missing)} required files. See PUSH_RSI_UNIFORM_GUIDE.md.')
        for path in missing[:12]:print('  '+path)
        raise SystemExit(1)
    source=args.source if args.source.is_absolute() else root/args.source
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!='f5b1637716611c673dd33f3dd894bf081a7dd88b7025943505d8287eba5ba13f':
        raise SystemExit('Source checkpoint differs from the validated Stage1V2Fixed_00078000.pth: '+digest)
    print(json.dumps(dict(required_files=len(expected),missing=0,source_sha256=digest),indent=2))


if __name__=='__main__':main()
