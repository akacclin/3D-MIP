import os
import pydicom
import numpy as np
import torch
import torch.nn.functional as F
import kornia.geometry.transform as T
from PIL import Image
import sys
import warnings
import traceback

def resample_volume(volume_np, original_spacing, target_spacing):
    """
    Resamples a 3D numpy volume (D, H, W) to the target spacing using PyTorch on GPU.
    Assumes spacing is (z, y, x).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Resampling using device: {device}")

    current_size = np.array(volume_np.shape) # (D, H, W)
    original_spacing_np = np.array(original_spacing) # (z, y, x)
    target_spacing_np = np.array(target_spacing) # (z, y, x)

    # Calculate target size based on original size and spacing ratios
    target_size = np.round(current_size * (original_spacing_np / target_spacing_np)).astype(int)
    target_size[target_size < 1] = 1 # Ensure dimensions are at least 1

    print(f"Original volume shape (z, y, x): {volume_np.shape}")
    print(f"Original spacing (z, y, x): {original_spacing} mm")
    print(f"Target spacing (z, y, x): {target_spacing} mm")
    print(f"Resampled target shape (z, y, x): {target_size}")

    # Convert numpy to torch tensor (B=1, C=1, D, H, W) and move to device
    # Convert to float32 as PyTorch might not support uint16 directly and it's needed for interpolation
    volume_ts = torch.from_numpy(volume_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    # Check if resampling is needed (allow small differences)
    if np.allclose(current_size, target_size, atol=1):
        print("Original volume is already close to target size, skipping resampling.")
        return volume_ts

    # Perform resampling using torch.nn.functional.interpolate (trilinear for 3D)
    try:
        resampled_volume_ts = F.interpolate(
            volume_ts,
            size=(int(target_size[0]), int(target_size[1]), int(target_size[2])), # (D', H', W') tuple
            mode='trilinear',
            align_corners=True
        )
        print(f"Resampling successful, shape on {device}: {resampled_volume_ts.shape[2:]}")

    except Exception as e:
         print(f"Resampling failed: {e}")
         traceback.print_exc()
         raise

    return resampled_volume_ts

def read_dicom_volume_with_info(folder_path):
    """
    Reads DICOM files from a folder, sorts them, and stacks into a 3D numpy array.
    Returns volume data (D, H, W) and spacing (z, y, x).
    """
    print(f"Reading DICOM files from: {folder_path}")
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]

    if not dicom_files:
        raise FileNotFoundError(f"No .dcm files found in {folder_path}")

    slices = []
    spacing_info = None # Store (SliceThickness, PixelSpacing[0], PixelSpacing[1]) for (z, y, x)
    unique_spacings = set()

    for dcm_file in dicom_files:
        try:
            ds = pydicom.dcmread(dcm_file)

            if not hasattr(ds, 'PixelData'):
                 continue # Skip non-image files

            slice_location = getattr(ds, 'SliceLocation', None)
            instance_number = getattr(ds, 'InstanceNumber', None)

            # Sorting slices: Prioritize SliceLocation, fallback to InstanceNumber
            if slice_location is not None:
                 slices.append((slice_location, ds))
            elif instance_number is not None:
                 slices.append((instance_number, ds))
            else:
                 warnings.warn(f"File {dcm_file} lacks common sorting tags (SliceLocation, InstanceNumber), skipping.")
                 continue

            # Capture spacing info from the first valid slice found
            slice_thickness = getattr(ds, 'SliceThickness', None)
            pixel_spacing = getattr(ds, 'PixelSpacing', None) # (row, col) -> (y, x)
            if slice_thickness is not None and pixel_spacing is not None and len(pixel_spacing) == 2:
                 current_spacing = (float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1]))
                 unique_spacings.add(tuple(current_spacing))
                 if spacing_info is None:
                     spacing_info = list(current_spacing)

        except pydicom.errors.InvalidDicomError:
             continue # Skip invalid DICOM files
        except Exception as e:
            print(f"Error reading file {dcm_file}: {e}")
            continue

    if not slices:
         raise FileNotFoundError(f"No valid DICOM image slices with sorting tags found in {folder_path}")

    slices.sort(key=lambda x: x[0])
    print(f"Successfully read and sorted {len(slices)} slices.")

    # Check spacing consistency
    if len(unique_spacings) > 1:
        print(f"Warning: Detected {len(unique_spacings)} different spacing infos. Using the first found spacing ({spacing_info}) for resampling. Recommend checking DICOM consistency.")
    elif len(unique_spacings) == 0:
         warnings.warn("No spacing information found. Assuming isotropic voxels with spacing [1, 1, 1]. This may cause distortion.")
         spacing_info = [1.0, 1.0, 1.0]
    else:
        if spacing_info:
            print(f"Pixel Spacing (y, x): [{spacing_info[1]:.6f}, {spacing_info[2]:.6f}] mm")
            print(f"Slice Thickness (z): {spacing_info[0]:.6f} mm")

    # Build volume
    try:
        ref_ds = slices[0][1]
        slice_shape = (ref_ds.Rows, ref_ds.Columns)
        pixel_dtype = ref_ds.pixel_array.dtype

        pixel_arrays = []
        for i, (sort_key, ds) in enumerate(slices):
             # Check if dimensions and dtype match the first slice for stacking
             if ds.Rows == slice_shape[0] and ds.Columns == slice_shape[1] and ds.pixel_array.dtype == pixel_dtype:
                 pixel_arrays.append(ds.pixel_array)
             else:
                 # Optionally warn about skipped slices if needed
                 # warnings.warn(f"Skipping slice {getattr(ds, 'InstanceNumber', i)} due to inconsistent shape or dtype.")
                 pass

        if not pixel_arrays:
             raise ValueError("No valid slices found for stacking (check consistency).")

        volume_data = np.stack(pixel_arrays, axis=0) # Stack along the z-axis

    except Exception as e:
        raise RuntimeError(f"Error processing pixel data: {e}")

    print(f"Original volume shape (z, y, x): {volume_data.shape}")
    print(f"Using original spacing (z, y, x): {spacing_info} mm")

    return volume_data, spacing_info


def generate_mip_rotations(volume_ts, output_folder, rotation_step_deg=10, rotation_axis='Y'):
    """
    Generates MIP images for 180 degree rotation using PyTorch on GPU.
    Assumes volume_ts is a (1, 1, D, H, W) tensor on the correct device,
    and has been resampled to isotropic voxels.
    Rotation_axis: 'X', 'Y', or 'Z'.
    """
    device = volume_ts.device
    print(f"Generating MIPs using device: {device}")

    _, _, D, H, W = volume_ts.shape
    print(f"Volume shape for rotation (D, H, W): ({D}, {H}, {W})")

    # Angles for rotation (0, 10, ..., 350 degrees)
    angles_deg = torch.arange(0, 190, rotation_step_deg, device=device, dtype=volume_ts.dtype)

    # Center of rotation (x, y, z) where x=W, y=H, z=D
    center = torch.tensor([[W/2, H/2, D/2]], device=device, dtype=volume_ts.dtype) # Shape (1, 3)

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Use tqdm for progress bar if available
    try:
        from tqdm import tqdm
        angles_iter = tqdm(angles_deg, desc=f"Generating MIPs (Rotating around {rotation_axis}-axis)")
    except ImportError:
        angles_iter = angles_deg
        print("Tip: Install tqdm (pip install tqdm) for a progress bar.")

    print(f"\nStarting {len(angles_deg)} MIP generations...")

    first_mip_max = None # For consistent scaling

    for i, angle_deg in enumerate(angles_iter):
        angle_value = angle_deg.item()

        # Prepare Euler angles (Yaw, Pitch, Roll) based on rotation_axis
        # Kornia's rotate3d likely uses Z-Y-X Euler angles (Yaw, Pitch, Roll)
        # Yaw: rotation around Z axis
        # Pitch: rotation around Y axis
        # Roll: rotation around X axis
        yaw_batch = torch.tensor([0.0], device=device, dtype=volume_ts.dtype)
        pitch_batch = torch.tensor([0.0], device=device, dtype=volume_ts.dtype)
        roll_batch = torch.tensor([0.0], device=device, dtype=volume_ts.dtype)

        if rotation_axis.upper() == 'Y':
            # Rotate around Y-axis (Pitch)
            pitch_batch = angle_deg.unsqueeze(0)
        elif rotation_axis.upper() == 'X':
            # Rotate around X-axis (Roll)
            roll_batch = angle_deg.unsqueeze(0)
        elif rotation_axis.upper() == 'Z':
             # Rotate around Z-axis (Yaw)
            yaw_batch = angle_deg.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported rotation_axis: {rotation_axis}. Choose 'X', 'Y', or 'Z'.")


        try:
            # Rotate the volume using Euler angles (degrees)
            rotated_volume_ts = T.rotate3d(
                volume_ts,           # (1, 1, D, H, W)
                yaw_batch,           # (1,) - Rotation around Z (degrees)
                pitch_batch,         # (1,) - Rotation around Y (degrees)
                roll_batch,          # (1,) - Rotation around X (degrees)
                center=center,       # (1, 3) in (cx, cy, cz) = (W/2, H/2, D/2) format
                mode='bilinear',     # Interpolation mode
                padding_mode='zeros' # Fill outside areas with zeros
            )

            # Perform Maximum Intensity Projection along Depth (dim 2)
            mip_ts, _ = torch.max(rotated_volume_ts, dim=2) # Result shape (1, 1, H, W)

            # Convert to numpy array (H, W)
            mip_np = mip_ts.squeeze().cpu().numpy()

            # Scale to 0-255 for saving as 8-bit image
            if first_mip_max is None:
                 first_mip_max = mip_np.max().item()
                 if first_mip_max < 1e-6:
                     first_mip_max = 1.0
                     warnings.warn("First MIP max value is zero or near zero, scaling by 1.0.")

            scaled_mip_np = (mip_np / first_mip_max) * 255.0
            scaled_mip_np = np.clip(scaled_mip_np, 0, 255).astype(np.uint8)

            # Save image
            img = Image.fromarray(scaled_mip_np)
            output_filename = os.path.join(output_folder, f"mip_{int(angle_value):03d}.jpg")
            img.save(output_filename)

        except Exception as e:
             print(f"\nError processing angle {angle_value}°: {e}")
             traceback.print_exc()
             continue # Continue to the next angle


    print("\nMIP image generation complete.")


if __name__ == "__main__":
    input_dicom_folder = r"" # Path to the folder containing DICOM files
    output_mip_folder = r"" # Path to save generated MIP images

    # Resampling settings (recommended for correct rotation)
    # Set target spacing to a specific value (e.g., 0.3 mm isotropic)
    target_spacing_value = None
    # Or set to None to use the minimum spacing from original data

    # Choose the rotation axis: 'X', 'Y', or 'Z'
    rotation_axis = 'Z' # 'Y' for vertical rotation (patient upright)
                        # 'X' for horizontal rotation (patient on side)
                        # 'Z' for axial rotation (like spinning a top)

    if not os.path.isdir(input_dicom_folder):
        print(f"Error: Input folder not found or is not a directory: {input_dicom_folder}")
        sys.exit(1)

    try:
        # Step 1: Read DICOM files and get volume data and spacing
        volume_data_np, original_spacing = read_dicom_volume_with_info(input_dicom_folder)

        # Determine target spacing
        if target_spacing_value is None:
            if original_spacing is None:
                 raise ValueError("Could not determine original spacing. Please specify target_spacing_value or ensure DICOMs have spacing tags.")
            min_spacing = min(original_spacing)
            target_spacing_xyz = [min_spacing, min_spacing, min_spacing]
            print(f"No target spacing value specified, using min original spacing {min_spacing:.3f} mm as target.")
        else:
             target_spacing_xyz = [target_spacing_value, target_spacing_value, target_spacing_value]
             print(f"Using specified target spacing {target_spacing_value:.3f} mm for resampling.")

        # Step 2: Resample to isotropic voxels (Crucial for correct rotation)
        volume_tensor_resampled = resample_volume(volume_data_np, original_spacing, target_spacing_xyz)

        # Step 3: Generate and save rotated MIP images
        generate_mip_rotations(volume_tensor_resampled, output_mip_folder, rotation_step_deg=10, rotation_axis=rotation_axis)

        print(f"All generated MIP images saved to folder: {output_mip_folder}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
