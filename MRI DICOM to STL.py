import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
from scipy import ndimage
from skimage.morphology import binary_closing, ball
from stl import mesh

# === CONFIGURATION ===
your_directory = "C:/Users/taral/Documents/vscode/data_and_image_files/mri"
initial_slice_name = "1-033.dcm"
slice_file_prefix = "1-"
slice_file_extension = ".dcm"
slices_sub_directory = ""

# === LOAD REFERENCE SLICE ===
initial_slice_path = os.path.join(your_directory, initial_slice_name)
RefDs = pydicom.dcmread(initial_slice_path)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns))
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
ArrayDicom[:, :] = RefDs.pixel_array

# === SLICE SELECTION VARIABLES ===
InFile = 100
FinalFile = 192
StartSlice = 123
EndSlice = 148

assert InFile <= StartSlice <= EndSlice <= FinalFile, "Slice range is outside bounds."

# === LOAD DICOM SLICES ===
SliceCount = EndSlice - StartSlice + 1
ArrayDicom3D = np.zeros((SliceCount, RefDs.Rows, RefDs.Columns), dtype=RefDs.pixel_array.dtype)

for i, ix in enumerate(range(StartSlice, EndSlice + 1)):
    slice_filename = f'{slice_file_prefix}{ix:03d}{slice_file_extension}'
    slice_path = os.path.join(your_directory, slices_sub_directory, slice_filename)
    RefDs = pydicom.dcmread(slice_path)
    ArrayDicom3D[i, :, :] = RefDs.pixel_array

# === SMOOTHING AND THRESHOLDING ===
smoothed3D = ndimage.gaussian_filter(ArrayDicom3D.astype(np.float32), sigma=1)
Threshold = 600
mask3D = (smoothed3D > Threshold).astype(np.uint8)

for ix in range(mask3D.shape[0]):
    mask3D[ix] = ndimage.binary_fill_holes(mask3D[ix]).astype(np.uint8)

mask3D = binary_closing(mask3D, ball(2))

# === ISOLATE CENTRAL TUMOR ===
labeled_array, num_features = ndimage.label(mask3D)
sizes = ndimage.sum(mask3D, labeled_array, range(num_features + 1))

center = np.array(mask3D.shape) // 2
max_size = 0
best_label = 0

for label in range(1, num_features + 1):
    coords = np.argwhere(labeled_array == label)
    if len(coords) == 0:
        continue
    centroid = coords.mean(axis=0)
    distance = np.linalg.norm(centroid - center)
    size = sizes[label]
    if size > max_size and distance < 80:
        max_size = size
        best_label = label

tumor_mask = (labeled_array == best_label).astype(np.uint8)

# === FILL INTERNAL 3D GAPS ===
def fill_3d_holes(binary_mask):
    inverse = 1 - binary_mask
    filled = ndimage.binary_fill_holes(inverse)
    internal_holes = filled ^ inverse
    return binary_mask | internal_holes

tumor_mask = fill_3d_holes(tumor_mask).astype(np.uint8)

# === FILLED VOLUME (for display or scalar mesh) ===
SegmentedArrayDicom3D = tumor_mask * ArrayDicom3D

# === VOXELIZED STL EXPORT (Watertight) ===
def voxel_to_mesh(tumor_mask, filename="tumor_filled_voxel.stl", voxel_size=1.0):
    """Create a watertight STL mesh by treating each voxel as a cube."""
    from stl import mesh
    filled_voxels = np.argwhere(tumor_mask == 1)

    # Cube vertex offsets for each voxel
    cube_verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ]) * voxel_size

    # Faces of the cube (triangles)
    faces = np.array([
        [0, 3, 1], [1, 3, 2],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
        [4, 5, 6], [4, 6, 7]
    ])

    total_triangles = filled_voxels.shape[0] * faces.shape[0]
    tumor_mesh = mesh.Mesh(np.zeros(total_triangles, dtype=mesh.Mesh.dtype))

    for i, voxel in enumerate(filled_voxels):
        base_idx = i * faces.shape[0]
        cube = cube_verts + voxel[::-1] * voxel_size  # reverse ZYX to XYZ
        for j in range(faces.shape[0]):
            for k in range(3):
                tumor_mesh.vectors[base_idx + j][k] = cube[faces[j][k]]

    tumor_mesh.save(filename)
    print(f"Voxelized STL saved: {filename}")

voxel_to_mesh(tumor_mask, filename="tumor_filled_voxel.stl")

# === OPTIONAL: View a middle slice ===
mid_slice = tumor_mask.shape[0] // 2
plt.imshow(SegmentedArrayDicom3D[mid_slice], cmap="gray")
plt.title("Mid Slice - Tumor Only (Filled)")
plt.show()