"""Create a standalone browser viewer of measured simulation traces (no network)."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parent

def build(folder):
    meta=json.loads((folder/'replay.json').read_text())
    req=json.loads((folder/'request.json').read_text())
    schedule=json.loads((folder/'schedule.json').read_text())
    release=json.loads((folder/'release_audit.json').read_text())
    with np.load(folder/'rollout.npz') as z:
        arrays={k:np.round(z[k],4).tolist() if np.issubdtype(z[k].dtype,np.floating) else z[k].tolist() for k in z.files}
    names=meta['body_names']; links=[]
    xml=ET.parse(ROOT/'teacher_runtime/tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml')
    def visit(node,parent=None):
        name=node.get('name'); current=names.index(name) if name in names else parent
        if current is not None and parent is not None and current!=parent: links.append([parent,current])
        for child in node.findall('body'): visit(child,current)
    for body in xml.getroot().find('worldbody').findall('body'): visit(body)
    data=dict(meta=meta,scene=req['current_scene'],schedule=schedule['schedule'],release=release,trace=arrays,links=links)
    template=(ROOT/'prompts/replay_view.html').read_text()
    text=template.replace('__REPLAY_DATA__',json.dumps(data,separators=(',',':')).replace('</','<\\/'))
    (folder/'view.html').write_text(text)
    print(folder/'view.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);build(p.parse_args().folder)
