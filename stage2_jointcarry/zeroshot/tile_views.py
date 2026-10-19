"""Arrange the two Isaac Gym windows on the existing VNC display; no processes stopped."""
import ctypes as C
x=C.CDLL('libX11.so.6'); W=C.c_ulong; P=C.c_void_p
x.XOpenDisplay.argtypes=[C.c_char_p];x.XOpenDisplay.restype=P
x.XDefaultRootWindow.argtypes=[P];x.XDefaultRootWindow.restype=W
x.XQueryTree.argtypes=[P,W,C.POINTER(W),C.POINTER(W),C.POINTER(C.POINTER(W)),C.POINTER(C.c_uint)]
x.XFetchName.argtypes=[P,W,C.POINTER(C.c_char_p)]
x.XFree.argtypes=[P]
x.XMoveResizeWindow.argtypes=[P,W,C.c_int,C.c_int,C.c_uint,C.c_uint]
x.XStoreName.argtypes=[P,W,C.c_char_p]
x.XFlush.argtypes=[P]
x.XCloseDisplay.argtypes=[P]
d=x.XOpenDisplay(b':100');assert d
root=x.XDefaultRootWindow(d);r=W();parent=W();children=C.POINTER(W)();count=C.c_uint()
x.XQueryTree(d,root,C.byref(r),C.byref(parent),C.byref(children),C.byref(count))
wins=[]
for i in range(count.value):
 name=C.c_char_p();x.XFetchName(d,children[i],C.byref(name))
 if name.value:
  title=name.value.decode(errors='replace')
  print(hex(children[i]),title)
  if 'Isaac Gym' in title or title in ['TABLE - zero shot','BOX - baseline']: wins.append(children[i])
  x.XFree(C.cast(name,P))
x.XFree(C.cast(children,P))
assert len(wins)==2, 'Expected exactly two viewer windows; left untouched'
# The table was started first and has the lower X resource ID.
for win,left,title in [(min(wins),800,b'TABLE - zero shot'),(max(wins),0,b'BOX - baseline')]:
 x.XMoveResizeWindow(d,win,left,0,800,900);x.XStoreName(d,win,title)
x.XFlush(d);x.XCloseDisplay(d)
