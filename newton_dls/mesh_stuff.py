from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr


CenterMode = Literal["origin", "com", "centroid"]


def _as_np(x, dtype=None) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _vec3(x, *, name: str) -> np.ndarray:
    arr = _as_np(x, np.float32)
    if arr is None:
        raise ValueError(f"{name} is None")
    arr = arr.reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must have 3 values, got shape {arr.shape}")
    return arr


def _vertices_array(vertices) -> np.ndarray:
    arr = _as_np(vertices, np.float32)
    if arr is None or arr.size == 0:
        raise ValueError("mesh.vertices is empty")
    return arr.reshape(-1, 3)


def _triangle_indices_array(indices) -> np.ndarray:
    arr = _as_np(indices, np.int32)
    if arr is None or arr.size == 0:
        raise ValueError("mesh.indices is empty")
    arr = arr.reshape(-1)
    if arr.size % 3 != 0:
        raise ValueError(f"mesh.indices length must be divisible by 3, got {arr.size}")
    return arr.reshape(-1, 3)


def _optional_normals(normals, vertex_count: int) -> np.ndarray | None:
    if normals is None:
        return None
    arr = _as_np(normals, np.float32)
    if arr is None or arr.size == 0:
        return None
    arr = arr.reshape(-1, 3)
    if arr.shape[0] != vertex_count:
        return None
    return arr


def _optional_uvs(uvs, vertex_count: int) -> np.ndarray | None:
    if uvs is None:
        return None
    arr = _as_np(uvs, np.float32)
    if arr is None or arr.size == 0:
        return None
    arr = arr.reshape(-1, 2)
    if arr.shape[0] != vertex_count:
        return None
    return arr


def _rgba255(color) -> np.ndarray | None:
    if color is None:
        return None

    arr = _as_np(color, np.float32)
    if arr is None:
        return None

    arr = arr.reshape(-1)
    if arr.size == 3:
        arr = np.concatenate([arr, [1.0 if np.max(arr) <= 1.0 else 255.0]])
    elif arr.size != 4:
        raise ValueError(f"mesh.color must have 3 or 4 values, got {arr.size}")

    if np.max(arr) <= 1.0:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    else:
        arr = np.clip(arr, 0.0, 255.0)

    return arr.astype(np.uint8)


def _texture_to_image(texture) -> np.ndarray | None:
    if texture is None:
        return None

    # newton.Mesh.texture may be a path or an image array
    if isinstance(texture, (str, Path)):
        path = Path(texture)
        if not path.exists():
            return None
        try:
            from PIL import Image
        except ImportError:
            return None
        img = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
        return img

    img = np.asarray(texture)

    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)

    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        return None

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            if np.nanmax(img) <= 1.0:
                img = np.clip(img, 0.0, 1.0) * 255.0
            else:
                img = np.clip(img, 0.0, 255.0)
        else:
            img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

    return img


def _center_offset(vertices: np.ndarray, mesh, center: CenterMode) -> np.ndarray:
    if center == "origin":
        return np.zeros(3, dtype=np.float32)
    if center == "centroid":
        return vertices.mean(axis=0, dtype=np.float32)
    if center == "com":
        if not hasattr(mesh, "com"):
            raise ValueError("center='com' requested but mesh has no .com")
        return _vec3(mesh.com, name="mesh.com")
    raise ValueError(f"Unknown center mode: {center}")

def compute_vertex_normals(vertices, indices, *, flip=False, eps=1e-12):
    """
    Compute area-weighted per-vertex normals from a triangle mesh.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
    indices : array-like, shape (M, 3) or flat length 3*M
    flip : bool
        Flip all normals. Useful if triangle winding is reversed.
    eps : float
        Small threshold to avoid division by zero.

    Returns
    -------
    normals : np.ndarray, shape (N, 3), dtype float32
    """
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tri = np.asarray(indices, dtype=np.int32).reshape(-1)
    if tri.size % 3 != 0:
        raise ValueError(f"indices length must be divisible by 3, got {tri.size}")
    tri = tri.reshape(-1, 3)

    normals = np.zeros_like(v, dtype=np.float32)

    # Triangle vertices
    p0 = v[tri[:, 0]]
    p1 = v[tri[:, 1]]
    p2 = v[tri[:, 2]]

    # Area-weighted face normals
    if flip:
        face_normals = np.cross(p2 - p0, p1 - p0)
    else:
        face_normals = np.cross(p1 - p0, p2 - p0)

    # Accumulate onto vertices
    np.add.at(normals, tri[:, 0], face_normals)
    np.add.at(normals, tri[:, 1], face_normals)
    np.add.at(normals, tri[:, 2], face_normals)

    # Normalize
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    good = lengths[:, 0] > eps
    normals[good] /= lengths[good]

    # Fallback for isolated/degenerate vertices
    if not np.all(good):
        centroid = v.mean(axis=0, keepdims=True)
        fallback = v - centroid
        fb_len = np.linalg.norm(fallback, axis=1, keepdims=True)
        fb_good = fb_len[:, 0] > eps
        fallback[fb_good] /= fb_len[fb_good]
        normals[~good] = fallback[~good]

    return normals.astype(np.float32)

def newton_mesh_to_rerun(
    mesh,
    *,
    center: CenterMode = "origin",
) -> rr.Mesh3D:
    """
    Convert one newton.Mesh to one rr.Mesh3D.

    center:
        - "origin": preserve mesh local frame exactly
        - "com": shift vertices so the local frame becomes the mesh COM
        - "centroid": shift vertices so the local frame becomes the vertex centroid
    """
    vertices = _vertices_array(mesh.vertices)
    indices = _triangle_indices_array(mesh.indices)

    offset = _center_offset(vertices, mesh, center)
    vertices = vertices - offset[None, :]

    normals = _optional_normals(getattr(mesh, "normals", None), vertices.shape[0])
    normals = normals if normals is not None else compute_vertex_normals(vertices, indices)
    
    uvs = _optional_uvs(getattr(mesh, "uvs", None), vertices.shape[0])

    kwargs: dict = {
        "vertex_positions": vertices,
        "triangle_indices": indices,
    }

    if normals is not None:
        kwargs["vertex_normals"] = normals
    if uvs is not None:
        kwargs["vertex_texcoords"] = uvs

    color = _rgba255(getattr(mesh, "color", None))
    if color is not None:
        kwargs["albedo_factor"] = color

    texture = _texture_to_image(getattr(mesh, "texture", None))
    if texture is not None:
        kwargs["albedo_texture"] = texture

    # Newton roughness / metallic / inertia / sdf have no direct Mesh3D field.
    return rr.Mesh3D(**kwargs)
