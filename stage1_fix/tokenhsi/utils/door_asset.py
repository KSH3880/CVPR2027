"""Left-hinged, right-handled self-closing door, viewed from local -X.

Local +X points through the doorway, +Y is the viewer's left, +Z is up.
Positive hinge angle swings the right edge away (+X) and towards the left (+Y).
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np


@dataclass(frozen=True)
class DoorSpec:
    width: float = .9
    height: float = 2.1
    thickness: float = .045
    bottom_gap: float = .015
    frame_width: float = .08
    frame_depth: float = .12
    frame_gap: float = .03
    handle_inset: float = .09
    handle_height: float = 1.05
    handle_projection: float = .075
    handle_length: float = .13
    panel_mass: float = 12.
    max_angle_degrees: float = 110.
    spring_stiffness: float = 6.0  # N m / rad; sustained opening requires force.
    spring_damping: float = 3.0    # N m s / rad.

    def __post_init__(self):
        for key,value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Door parameter must be positive and finite: '+key)
        if self.frame_gap <= self.thickness/2:
            raise ValueError('Frame clearance must exceed half the leaf thickness')
        if not 0 < self.handle_inset < self.width/2:
            raise ValueError('Handle must lie near the right edge')
        if not self.bottom_gap < self.handle_height < self.height:
            raise ValueError('Handle height is outside the panel')
        if not 90 <= self.max_angle_degrees <= 150:
            raise ValueError('Door opening limit must be 90..150 degrees')
        if self.handle_projection <= self.thickness/2:
            raise ValueError('Grip must stand clear of the panel')

    @property
    def max_angle(self): return math.radians(self.max_angle_degrees)

    @property
    def hinge(self): return np.array([0.,self.width/2,0.])

    @property
    def front_handle_local(self):
        # Handle link origin is the graspable lever centre, relative to the hinge.
        return np.array([-self.handle_projection,-self.width+self.handle_inset,self.handle_height])

    def handle_position(self, angle, base=(0.,0.,0.), yaw=0.):
        c,s=math.cos(angle),math.sin(angle)
        rz=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        c,s=math.cos(yaw),math.sin(yaw)
        base_r=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        return np.asarray(base)+base_r@(self.hinge+rz@self.front_handle_local)

    def closing_torque(self, angle, velocity=0.):
        return -self.spring_stiffness*np.asarray(angle)-self.spring_damping*np.asarray(velocity)


def numbers(values): return ' '.join(f'{float(x):.9g}' for x in values)


def write_door_asset(path, spec=DoorSpec()):
    robot=ET.Element('robot',name='left_hinge_right_handle_door')
    for name,rgba in [('frame',(.17,.20,.25,1)),('wood',(.56,.32,.14,1)),('metal',(.8,.83,.86,1))]:
        material=ET.SubElement(robot,'material',name=name)
        ET.SubElement(material,'color',rgba=numbers(rgba))

    def inertial(link,mass,center,size):
        i=ET.SubElement(link,'inertial')
        ET.SubElement(i,'origin',xyz=numbers(center),rpy='0 0 0')
        ET.SubElement(i,'mass',value=str(mass))
        x,y,z=size
        ET.SubElement(i,'inertia',ixx=str(mass*(y*y+z*z)/12),iyy=str(mass*(x*x+z*z)/12),
                      izz=str(mass*(x*x+y*y)/12),ixy='0',ixz='0',iyz='0')

    def box(link,name,center,size,material):
        for kind in ('visual','collision'):
            item=ET.SubElement(link,kind,name=name+'_'+kind)
            ET.SubElement(item,'origin',xyz=numbers(center),rpy='0 0 0')
            geometry=ET.SubElement(item,'geometry')
            ET.SubElement(geometry,'box',size=numbers(size))
            if kind=='visual':ET.SubElement(item,'material',name=material)

    frame=ET.SubElement(robot,'link',name='frame')
    inertial(frame,1.,(0,0,0),(.1,.1,.1))
    frame_h=spec.height+spec.bottom_gap+spec.frame_gap
    side=spec.width/2+spec.frame_gap+spec.frame_width/2
    for name,y in [('left_post',side),('right_post',-side)]:
        box(frame,name,(0,y,frame_h/2),(spec.frame_depth,spec.frame_width,frame_h),'frame')
    box(frame,'lintel',(0,0,frame_h+spec.frame_width/2),
        (spec.frame_depth,2*side+spec.frame_width,spec.frame_width),'frame')
    panel=ET.SubElement(robot,'link',name='panel')
    center=(0,-spec.width/2,spec.height/2+spec.bottom_gap)
    size=(spec.thickness,spec.width,spec.height)
    inertial(panel,spec.panel_mass,center,size)
    box(panel,'leaf',center,size,'wood')
    hinge=ET.SubElement(robot,'joint',name='door_hinge',type='revolute')
    ET.SubElement(hinge,'parent',link='frame');ET.SubElement(hinge,'child',link='panel')
    ET.SubElement(hinge,'origin',xyz=numbers(spec.hinge),rpy='0 0 0')
    ET.SubElement(hinge,'axis',xyz='0 0 1')
    ET.SubElement(hinge,'limit',lower='0',upper=str(spec.max_angle),effort='40',velocity='4')
    ET.SubElement(hinge,'dynamics',damping='0',friction='0')
    for name,sign in [('handle',-1),('handle_back',1)]:
        link=ET.SubElement(robot,'link',name=name)
        inertial(link,.15,(0,0,0),(.02,spec.handle_length,.025))
        box(link,'lever',(0,0,0),(.024,spec.handle_length,.025),'metal')
        stem_length=spec.handle_projection-spec.thickness/2
        box(link,'stem',(-sign*stem_length/2,-spec.handle_length*.35,0),
            (stem_length,.025,.025),'metal')
        joint=ET.SubElement(robot,'joint',name=name+'_mount',type='fixed')
        ET.SubElement(joint,'parent',link='panel');ET.SubElement(joint,'child',link=name)
        offset=spec.front_handle_local.copy();offset[0]=sign*spec.handle_projection
        ET.SubElement(joint,'origin',xyz=numbers(offset),rpy='0 0 0')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(minidom.parseString(ET.tostring(robot)).toprettyxml(indent='  '))
    return path
