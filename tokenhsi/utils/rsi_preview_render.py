"""CPU triangle depth rendering of the actual MJCF collision surfaces and foot OBJ."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def rotate(q, points):
    cross=2*np.cross(q[:3],points)
    return points+q[3]*cross+np.cross(q[:3],cross)


def rounded_mesh(start,end,radius,sectors=10):
    """Capsule, including both hemispheres; zero length gives a sphere."""
    start,end=np.asarray(start),np.asarray(end)
    axis=end-start;length=np.linalg.norm(axis)
    axis=axis/length if length>1e-10 else np.array([0.,0.,1.])
    seed=np.array([1.,0.,0.]) if abs(axis[0])<.8 else np.array([0.,1.,0.])
    u=np.cross(axis,seed);u/=np.linalg.norm(u);v=np.cross(axis,u)
    rings=[]
    for center,angles in ((start,np.linspace(-np.pi/2,0,4)),(end,np.linspace(0,np.pi/2,4))):
        for angle in angles:
            phi=np.arange(sectors)*2*np.pi/sectors
            rings.extend(center+radius*(np.sin(angle)*axis+np.cos(angle)*(np.cos(phi)[:,None]*u+np.sin(phi)[:,None]*v)))
    faces=[]
    for ring in range(7):
        for i in range(sectors):
            a=ring*sectors+i;b=ring*sectors+(i+1)%sectors;c=a+sectors;d=b+sectors
            faces.extend(((a,b,c),(b,d,c)))
    return np.asarray(rings),np.asarray(faces)


def raster_triangle(pixels,depth,xyz,color):
    """Larger depth is nearer the eye; supports both triangle winding orders."""
    height,width=depth.shape
    lo=np.maximum(np.floor(xyz[:,:2].min(0)).astype(int),[0,0])
    hi=np.minimum(np.ceil(xyz[:,:2].max(0)).astype(int),[width-1,height-1])
    if np.any(lo>hi):return
    a,b,c=xyz
    denom=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
    if abs(denom)<1e-8:return
    y,x=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];x=x+.5;y=y+.5
    wa=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/denom
    wb=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/denom
    wc=1-wa-wb;z=wa*a[2]+wb*b[2]+wc*c[2]
    crop=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
    visible=(wa>=-1e-7)&(wb>=-1e-7)&(wc>=-1e-7)&(z>crop)
    crop[visible]=z[visible]
    pixels[lo[1]:hi[1]+1,lo[0]:hi[0]+1][visible]=color


class PreviewRenderer:
    def __init__(self,asset,names):
        asset=Path(asset);tree=ET.parse(asset)
        meshes={m.get('name'):asset.parent/m.get('file') for m in tree.findall('./asset/mesh')}
        self.parts=[];self.foot_parts=[]
        for body in tree.findall('.//worldbody//body'):
            name=body.get('name');bid=names.index(name)
            for g in body.findall('./geom'):
                if g.get('type')=='mesh':
                    vertices=[];faces=[]
                    for line in meshes[g.get('mesh')].read_text().splitlines():
                        if line.startswith('v '):vertices.append([float(v) for v in line.split()[1:4]])
                        elif line.startswith('f '):
                            polygon=[int(i.split('/')[0])-1 for i in line.split()[1:]]
                            faces.extend((polygon[0],polygon[j],polygon[j+1]) for j in range(1,len(polygon)-1))
                    vertices=np.asarray(vertices)+np.fromstring(g.get('pos','0 0 0'),sep=' ');faces=np.asarray(faces)
                    self.foot_parts.append(len(self.parts))
                else:
                    radius=float(g.get('size').split()[0])
                    if 'fromto' in g.attrib:
                        ends=np.fromstring(g.get('fromto'),sep=' ').reshape(2,3)
                    else:
                        center=np.fromstring(g.get('pos','0 0 0'),sep=' ');ends=np.stack((center,center))
                    vertices,faces=rounded_mesh(ends[0],ends[1],radius)
                color=[77,186,210] if name.startswith('right') else [106,214,175]
                if name in ('right_hand','left_hand'):color=[252,176,70]
                self.parts.append((bid,vertices,faces,np.array(color)))
        self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)

    def scene(self,body,rot,box,dims):
        parts=[]
        for bid,vertices,faces,color in self.parts:
            parts.append((body[bid]+rotate(rot[bid],vertices),faces,color))
        signs=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
        corners=box[:3]+rotate(box[3:7],signs*np.asarray(dims)/2)
        faces=np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
                        [1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]])
        parts.append((corners,faces,np.array([183,140,65])))
        return parts

    def panel(self,body,rot,box,dims,side=False,width=640,height=500):
        parts=self.scene(body,rot,box,dims)
        direction=box[:2]-body[0,:2];direction/=np.linalg.norm(direction)
        forward=np.r_[direction,0.];tangent=np.array([-forward[1],forward[0],0.])
        if side:
            right=forward;up=np.array([0.,0.,1.]);eye=-tangent
        else:
            # Both cameras stay on the human side instead of looking through the box.
            eye=-forward*.82+tangent*.57;eye=eye/np.linalg.norm(eye)
            right=np.cross(np.array([0.,0.,1.]),eye);right/=np.linalg.norm(right)
            eye=eye*np.cos(.27)+np.array([0.,0.,1.])*np.sin(.27)
            up=np.cross(eye,right)
        allpoints=np.concatenate([p for p,_,_ in parts]);ground=allpoints.copy();ground[:,2]=0
        bounds=np.concatenate((allpoints,ground))
        x=bounds@right;y=bounds@up
        scale=min((width-88)/max(np.ptp(x),1.),(height-100)/max(np.ptp(y),1.))
        center_x=(x.min()+x.max())/2;center_y=(y.min()+y.max())/2
        def project(points):
            return np.stack((width/2+(points@right-center_x)*scale,
                height/2-(points@up-center_y)*scale,points@eye),-1)
        pixels=np.zeros((height,width,3),dtype=np.uint8);pixels[:]=[19,27,38]
        depth=np.full((height,width),-np.inf)
        if not side:
            center=box[:2]/2+body[0,:2]/2
            for ix in range(-6,7):
                for iy in range(-6,7):
                    square=np.array([[ix,iy,0],[ix+1,iy,0],[ix+1,iy+1,0],[ix,iy+1,0]],dtype=float)
                    square[:,:2]=square[:,:2]*.5+center
                    color=[32,43,52] if (ix+iy)%2 else [38,50,59]
                    projected=project(square)
                    for face in ((0,1,2),(0,2,3)):raster_triangle(pixels,depth,projected[list(face)],color)
        light=np.array([-.3,-.4,1.]);light/=np.linalg.norm(light)
        for vertices,faces,color in parts:
            screen=project(vertices)
            for face in faces:
                triangle=vertices[face];normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
                normal/=max(np.linalg.norm(normal),1e-10)
                shade=.55+.45*abs(normal@light)
                raster_triangle(pixels,depth,screen[face],np.clip(color*shade,0,255).astype(np.uint8))
        im=Image.fromarray(pixels);draw=ImageDraw.Draw(im)
        if side:
            floor_y=float(project(np.array([[0.,0.,0.]]))[0,1])
            draw.line([(12,floor_y),(width-12,floor_y)],fill='#c7d3db',width=2)
            draw.text((18,floor_y+7),'GROUND z = 0',font=self.font,fill='#c7d3db')
            for index in self.foot_parts:
                points=parts[index][0];sole=points[np.argmin(points[:,2])];bottom=sole.copy();bottom[2]=0
                p=project(np.stack((sole,bottom)))
                draw.line([tuple(p[0,:2]),tuple(p[1,:2])],fill='#ffd674',width=3)
                draw.ellipse((p[1,0]-4,p[1,1]-4,p[1,0]+4,p[1,1]+4),fill='#ffd674')
        draw.text((16,12),'TRUE SIDE | world Z vertical' if side else 'OBLIQUE | opaque surfaces / depth',font=self.font,fill='#e3edf0')
        return im
