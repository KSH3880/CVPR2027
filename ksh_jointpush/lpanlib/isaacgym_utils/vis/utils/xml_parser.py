import os
import trimesh
import numpy as np
import xml.dom.minidom

def create_sphere(pos, size, MESH_SIMPLIFY=True):
    if pos == '':
        pos = [0, 0, 0]
    else:
        pos = [float(x) for x in pos.split(' ')]
    R = np.identity(4)
    R[:3, 3] = np.array(pos).T
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(size))
    mesh.apply_transform(R)

    if MESH_SIMPLIFY:
        face_count = 50
    else:
        face_count = 5000

    return mesh.simplify_quadric_decimation(face_count)

def create_capsule(from_to, size, MESH_SIMPLIFY=True):
    from_to = [float(x) for x in from_to.split(' ')]
    start_point = np.array(from_to[:3])
    end_point = np.array(from_to[3:])

    # 计算pos
    pos = (start_point + end_point) / 2.0

    # 计算rot
    # 用罗德里格公式, 由向量vec2求旋转矩阵
    vec1 = np.array([0, 0, 1.0])
    vec2 = (start_point - end_point)
    height = np.linalg.norm(vec2)
    vec2 = vec2 / np.linalg.norm(vec2)
    if vec2[2] != 1.0: # (如果方向相同时, 公式不适用, 所以需要判断一下)
        i = np.identity(3)
        v = np.cross(vec1, vec2)
        v_mat = [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
        s = np.linalg.norm(v)
        c = np.dot(vec1, vec2)
        R_mat = i + v_mat + np.matmul(v_mat, v_mat) * (1 - c) / (s * s)
    else:
        R_mat = np.identity(3)

    # 做transform
    T = np.identity(4)
    T[0:3, 0:3] = R_mat
    T[0:3, 3] = pos.T
    mesh = trimesh.creation.capsule(height, float(size))
    mesh.apply_transform(T)

    if MESH_SIMPLIFY:
        face_count = 50
    else:
        face_count = 1000

    return mesh.simplify_quadric_decimation(face_count)

def create_box(pos, size, MESH_SIMPLIFY=True):
    if pos == '':
        pos = [0, 0, 0]
    else:
        pos = [float(x) for x in pos.split(' ')]
    
    size = [float(x) * 2 for x in size.split(' ')]
    
    R = np.identity(4)
    R[:3, 3] = np.array(pos).T
    mesh = trimesh.creation.box(size)
    mesh.apply_transform(R)

    if MESH_SIMPLIFY:
        face_count = 50
    else:
        face_count = 1000

    return mesh.simplify_quadric_decimation(face_count)

def parse_geom_elements_from_xml(xml_path, MESH_SIMPLIFY=True): # only support box, sphere, mesh, and capsule (fromto format)
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement

    # support mesh type rigid body
    geoms = {}
    for info in root.getElementsByTagName('mesh'):
        name = info.getAttribute("name")
        file_path = os.path.join(os.path.dirname(xml_path), info.getAttribute("file"))
        geoms[name] = trimesh.load(file_path, process=False)

    body = root.getElementsByTagName('body')
    body_names = []
    body_meshes = []
    for b in body:
        name = b.getAttribute('name')
        child = b.childNodes

        mesh = []
        for c in child:
            if c.nodeType == 1:
                if c.nodeName == 'geom':
                    if c.getAttribute('type') == 'sphere':
                        size = c.getAttribute('size')
                        pos = c.getAttribute('pos')
                        mesh.append(create_sphere(pos, size, MESH_SIMPLIFY))
                    elif c.getAttribute('type') == 'box':
                        pos = c.getAttribute('pos')
                        size = c.getAttribute('size')
                        mesh.append(create_box(pos, size, MESH_SIMPLIFY))
                    elif c.getAttribute('type') == 'mesh':
                        key = c.getAttribute('mesh')
                        mesh.append(geoms[key])
                    else:
                        from_to = c.getAttribute('fromto')
                        size = c.getAttribute('size')
                        mesh.append(create_capsule(from_to, size, MESH_SIMPLIFY))
        mesh = trimesh.util.concatenate(mesh)

        body_names.append(name)
        body_meshes.append(mesh)
    
    return body_names, body_meshes
