"""Viewer-only initial/current heading overlays."""
import math
import numpy as np


def overlay_lines(goal, size, box_yaw, initial_roots, current_roots):
    lines=[];colors=[]
    def line(a,b,color):lines.append([a,b]);colors.append(color)
    for roots,initial in ((initial_roots,True),(current_roots,False)):
        for agent,root in enumerate(roots):
            x,y,z,w=root[3:7]
            yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            d=np.array([math.cos(yaw),math.sin(yaw),0.])
            t=np.array([-d[1],d[0],0.])
            start=np.array([root[0],root[1],.1 if initial else .16])
            end=start+.8*d
            color=(.6,.6,.6) if initial else ((.1,.8,1.),(1.,.2,.7),(.2,1.,.3))[agent%3]
            line(start,end,color)
            line(end,end-.2*d+.12*t,color);line(end,end-.2*d-.12*t,color)
    return np.asarray(lines,dtype=np.float32),np.asarray(colors,dtype=np.float32)
