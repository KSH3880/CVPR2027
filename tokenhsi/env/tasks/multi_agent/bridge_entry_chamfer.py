from pathlib import Path


WIDTH = .2
DEPTH = .4


def evaluation_enabled(value, is_eval):
    if value not in ('0', '1'):
        raise ValueError('BRIDGE_ENTRY_CHAMFER must be 0 or 1')
    if value == '1' and not is_eval:
        raise ValueError('BRIDGE_ENTRY_CHAMFER is evaluation-only')
    return value == '1'


def footprint(size, side):
    x, y = size[0]/2, size[1]/2
    polygon = [(-x,-y), (x-WIDTH,-y), (x,-y+DEPTH),
               (x,y-DEPTH), (x-WIDTH,y), (-x,y)]
    return polygon if side == 1 else [(-a,b) for a,b in reversed(polygon)]


def write_asset(directory, size, side):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = 'platform_'+'_'.join(f'{v:.6f}'.replace('.', 'p') for v in size)+f'_{side}'
    mesh, urdf = directory/(name+'.obj'), directory/(name+'.urdf')
    polygon = footprint(size, side)
    vertices = [(x,y,z) for z in (-size[2]/2,size[2]/2) for x,y in polygon]
    faces = []
    for i in range(1,5):
        faces += [(1,i+2,i+1), (7,i+7,i+8)]
    for i in range(6):
        a, b = i+1, (i+1)%6+1
        faces += [(a,b,b+6), (a,b+6,a+6)]
    mesh.write_text(''.join('v {:.8f} {:.8f} {:.8f}\n'.format(*v) for v in vertices)
                    + ''.join('f {} {} {}\n'.format(*f) for f in faces))
    shape = f'<geometry><mesh filename="{mesh.name}"/></geometry>'
    urdf.write_text(f'<robot name="entry_chamfer"><link name="platform">'
                    f'<visual>{shape}</visual><collision>{shape}</collision>'
                    '</link></robot>')
    return urdf
