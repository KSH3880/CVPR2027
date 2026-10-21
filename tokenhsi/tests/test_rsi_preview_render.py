"""Opaque occlusion must depend on depth, not the drawing order or winding."""
import numpy as np
from utils.rsi_preview_render import raster_triangle, rounded_mesh


def test_depth_occlusion_is_independent_of_draw_order():
    near=np.array([[2.,2.,2.],[28.,2.,2.],[2.,28.,2.]])
    far=near.copy();far[:,2]=1.
    results=[]
    for triangles in (((far,[200,150,30]),(near,[30,180,220])),
                      ((near[::-1],[30,180,220]),(far[::-1],[200,150,30]))):
        pixels=np.zeros((32,32,3),dtype=np.uint8);depth=np.full((32,32),-np.inf)
        for triangle,color in triangles:raster_triangle(pixels,depth,triangle,color)
        results.append(pixels)
    np.testing.assert_array_equal(results[0],results[1])
    np.testing.assert_array_equal(results[0][5,5],[30,180,220])


def test_capsule_includes_original_end_spheres():
    vertices,faces=rounded_mesh([0.,0.,0.],[0.,0.,1.],.1)
    assert abs(vertices[:,2].min()+.1)<1e-9
    assert abs(vertices[:,2].max()-1.1)<1e-9
    assert faces.min()>=0 and faces.max()<len(vertices)
