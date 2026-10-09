"""One-time, reproducible CC0 asset import; never runs during ROS startup.

Run from the repository root: python3 tools/import_marine_assets.py
Downloads only the pinned public URLs in sources.json. Source GLBs are hashed
in imported.json; cached downloads live outside the repository.
"""
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def export_visual(parts):
    """Deterministic Z-up COLLADA with the source's separate colour materials.

    The pinned marine pack uses solid PBR materials, not image textures.
    COLLADA's diffuse/specular model approximates PBR roughness in RViz.
    """
    root = ET.Element('COLLADA', xmlns='http://www.collada.org/2005/11/COLLADASchema', version='1.4.1')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'created').text = '2026-10-07T00:00:00Z'
    ET.SubElement(asset, 'modified').text = '2026-10-07T00:00:00Z'
    ET.SubElement(asset, 'unit', name='meter', meter='1')
    ET.SubElement(asset, 'up_axis').text = 'Z_UP'
    effects = ET.SubElement(root, 'library_effects')
    materials = ET.SubElement(root, 'library_materials')
    geometries = ET.SubElement(root, 'library_geometries')
    scenes = ET.SubElement(root, 'library_visual_scenes')
    scene = ET.SubElement(scenes, 'visual_scene', id='scene')
    for index, part in enumerate(parts):
        prefix = f'part{index}'
        material = part.visual.material
        if getattr(material, 'baseColorTexture', None) is not None:
            raise ValueError('Texture-bearing source needs explicit texture export')
        rgba = np.asarray(material.baseColorFactor, dtype=float) / 255.
        effect = ET.SubElement(effects, 'effect', id=prefix+'-effect')
        profile = ET.SubElement(effect, 'profile_COMMON')
        technique = ET.SubElement(profile, 'technique', sid='common')
        phong = ET.SubElement(technique, 'phong')
        ET.SubElement(ET.SubElement(phong, 'diffuse'), 'color').text = ' '.join(map(str, rgba))
        ET.SubElement(ET.SubElement(phong, 'specular'), 'color').text = '0.04 0.04 0.04 1'
        roughness = material.roughnessFactor or 1.
        ET.SubElement(ET.SubElement(phong, 'shininess'), 'float').text = str(max(0., 2 / roughness**2 - 2))
        mat = ET.SubElement(materials, 'material', id=prefix+'-material', name=material.name or prefix)
        ET.SubElement(mat, 'instance_effect', url='#'+prefix+'-effect')
        geometry = ET.SubElement(geometries, 'geometry', id=prefix)
        mesh = ET.SubElement(geometry, 'mesh')
        # Per-face normals preserve the original low-poly, flat-shaded appearance.
        vertices = part.vertices[part.faces].reshape(-1, 3)
        normals = np.repeat(part.face_normals, 3, axis=0)
        for label, values in [('positions', vertices), ('normals', normals)]:
            sid = prefix+'-'+label
            source = ET.SubElement(mesh, 'source', id=sid)
            ET.SubElement(source, 'float_array', id=sid+'-array', count=str(values.size)).text = ' '.join(format(v, '.9g') for v in values.ravel())
            accessor = ET.SubElement(ET.SubElement(source, 'technique_common'), 'accessor', source='#'+sid+'-array', count=str(len(values)), stride='3')
            for axis in 'XYZ':
                ET.SubElement(accessor, 'param', name=axis, type='float')
        vertex = ET.SubElement(mesh, 'vertices', id=prefix+'-vertices')
        ET.SubElement(vertex, 'input', semantic='POSITION', source='#'+prefix+'-positions')
        triangles = ET.SubElement(mesh, 'triangles', count=str(len(part.faces)), material='material')
        ET.SubElement(triangles, 'input', semantic='VERTEX', source='#'+prefix+'-vertices', offset='0')
        ET.SubElement(triangles, 'input', semantic='NORMAL', source='#'+prefix+'-normals', offset='0')
        ET.SubElement(triangles, 'p').text = ' '.join(map(str, range(len(vertices))))
        node = ET.SubElement(scene, 'node', id=prefix+'-node')
        instance = ET.SubElement(node, 'instance_geometry', url='#'+prefix)
        common = ET.SubElement(ET.SubElement(instance, 'bind_material'), 'technique_common')
        ET.SubElement(common, 'instance_material', symbol='material', target='#'+prefix+'-material')
    ET.SubElement(ET.SubElement(root, 'scene'), 'instance_visual_scene', url='#scene')
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def decode_normalized_glb(data):
    """Bake normalized integer attributes to floats before trimesh import.

    These GLBs use KHR_mesh_quantization. Trimesh reads integer POSITION values
    without their normalized factor, producing enormous, incorrect animals.
    """
    magic, version, _ = struct.unpack_from('<III', data)
    if magic != 0x46546C67 or version != 2:
        raise ValueError('Expected binary glTF 2')
    length, kind = struct.unpack_from('<II', data, 12)
    if kind != 0x4E4F534A:
        raise ValueError('Expected GLB JSON chunk')
    document = json.loads(data[20:20+length])
    offset = 20 + length
    binary_length, kind = struct.unpack_from('<II', data, offset)
    if kind != 0x004E4942:
        raise ValueError('Expected GLB binary chunk')
    binary = bytearray(data[offset+8:offset+8+binary_length])
    types = {5120: np.dtype('i1'), 5121: np.dtype('u1'),
             5122: np.dtype('<i2'), 5123: np.dtype('<u2')}
    widths = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    for accessor in document['accessors']:
        if not accessor.get('normalized', False):
            continue
        dtype = types[accessor['componentType']]
        width = widths[accessor['type']]
        view = document['bufferViews'][accessor['bufferView']]
        if view.get('buffer', 0) != 0 or 'sparse' in accessor:
            raise ValueError('Unsupported external/sparse quantized attribute')
        values = np.ndarray((accessor['count'], width), dtype=dtype, buffer=bytes(binary),
            offset=view.get('byteOffset', 0) + accessor.get('byteOffset', 0),
            strides=(view.get('byteStride', width*dtype.itemsize), dtype.itemsize))
        values = values.astype('<f4') / np.iinfo(dtype).max
        if dtype.kind == 'i':
            values = np.maximum(values, -1.)
        binary.extend(b'\0' * (-len(binary) % 4))
        new_view = {'buffer': 0, 'byteOffset': len(binary), 'byteLength': values.nbytes}
        binary.extend(values.astype('<f4').tobytes())
        accessor.update(bufferView=len(document['bufferViews']), byteOffset=0,
                        componentType=5126, normalized=False)
        if 'min' in accessor:
            accessor['min'] = values.min(axis=0).tolist()
            accessor['max'] = values.max(axis=0).tolist()
        document['bufferViews'].append(new_view)
    document['buffers'][0]['byteLength'] = len(binary)
    encoded = json.dumps(document, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary.extend(b'\0' * (-len(binary) % 4))
    return (struct.pack('<III', magic, version, 28+len(encoded)+len(binary))
            + struct.pack('<II', len(encoded), 0x4E4F534A) + encoded
            + struct.pack('<II', len(binary), 0x004E4942) + binary)


def group_for(name):
    if any(word in name for word in ('shark', 'hammerhead')):
        return 'Sharks'
    if any(word in name for word in ('whale', 'humpback', 'orca', 'porpoise', 'dugong', 'sea-lion', 'otter')):
        return 'Marine mammals'
    if 'ray' in name:
        return 'Rays'
    if 'turtle' in name:
        return 'Turtles'
    if any(word in name for word in ('octopus', 'squid', 'jellyfish', 'man-o-war', 'isopod')):
        return 'Other marine animals'
    return 'Fish'


def main():
    directory = Path(__file__).resolve().parents[1] / 'resource' / 'obstacle_meshes'
    sources = json.loads((directory / 'sources.json').read_text())
    records, catalog = [], {}
    (directory / 'marine').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='uvms-marine-import-') as cache:
        for asset in sources['assets']:
            path = Path(cache) / (asset['name'] + '.glb')
            subprocess.run(['curl', '-L', '--fail', '--max-time', '30', '--retry', '2',
                            '-sS', '-o', str(path), asset['url']], check=True)
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != asset['sha256']:
                raise ValueError(f'Source checksum mismatch: {asset["name"]}')
            scene = trimesh.load(io.BytesIO(decode_normalized_glb(data)), file_type='glb')
            parts = list(scene.dump())
            mesh = trimesh.util.concatenate(parts)
            # Right-handed Y-up/Z-longitudinal -> ROS Z-up/X-longitudinal.
            transform = np.eye(4)
            transform[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
            mesh.apply_transform(transform)
            transform[:3, 3] = -mesh.bounds.mean(axis=0)
            for part in parts:
                part.apply_transform(transform)
            mesh = trimesh.util.concatenate(parts)
            if not np.isfinite(mesh.vertices).all() or not 0.01 < max(mesh.extents) < 100:
                raise ValueError(f'Invalid physical bounds: {asset["name"]}: {mesh.extents}')
            output = directory / 'marine' / (asset['name'] + '.stl')
            mesh.export(output)
            visual = output.with_suffix('.dae')
            visual.write_bytes(export_visual(parts))
            group = group_for(asset['name'])
            length = 1. if group == 'Fish' else 2.
            scale = float(length / max(mesh.extents))
            label = f'{group} / {asset["name"].replace("-", " ").title()} ({length:g} m)'
            catalog[label] = {'collision_mesh_resource': f'package://simlab/obstacle_meshes/marine/{output.name}',
                              'collision_mesh_scale': [scale]*3,
                              'visual_mesh_resource': f'package://simlab/obstacle_meshes/marine/{visual.name}',
                              'visual_dimensions': [scale]*3}
            records.append({**asset, 'source_sha256': hashlib.sha256(data).hexdigest(),
                'stl_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'visual_sha256': hashlib.sha256(visual.read_bytes()).hexdigest(),
                'extents_m': mesh.extents.tolist(), 'triangles': len(mesh.faces)})
            print(asset['name'], len(mesh.faces), mesh.extents.round(3).tolist(), flush=True)
    (directory / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    (directory / 'imported.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
