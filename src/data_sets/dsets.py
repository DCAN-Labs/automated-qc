import copy
import functools
import logging
import os
import random

import torch
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
import nibabel as nib
import numpy as np
from scipy import ndimage


from util.disk import getCache

log = logging.getLogger(__name__)

raw_cache = getCache("auto-qc")

# Default spatial target for 3D anatomical (T1w/T2w) inputs. Kept for backwards
# compatibility with models 00-05 and model_02r*.
DEFAULT_3D_TARGET_SHAPE = (260, 320, 320)

# Default spatial target for 4D field map (spin-echo EPI) inputs. EPI field maps
# are acquired at far lower spatial resolution than the anatomicals, so reusing
# the 260x320x320 target would pad ~95% zeros and waste an enormous amount of
# memory once multiplied by the frame axis.
DEFAULT_4D_TARGET_SHAPE = (96, 96, 64)

# Default number of frames retained from a 4D field map acquisition.
DEFAULT_NUM_FRAMES = 4


@dataclass(frozen=True)
class ScanSpec:
    """Immutable description of how a scan should be loaded and preprocessed.

    This is passed into the disk-cache-memoized loader, so every field must be
    hashable. Including it in the cache key is deliberate: without it, changing
    the target shape or frame count while reusing an existing AUTO_QC_CACHE_DIR
    would silently return stale tensors of the previous shape.
    """

    input_mode: str = "3d"                  # "3d" or "4d"
    target_shape: Tuple[int, ...] = DEFAULT_3D_TARGET_SHAPE
    num_frames: Optional[int] = None        # None keeps every frame (4d only)
    frame_padding: str = "edge"             # "edge" or "zero"
    frame_selection: str = "center"         # "center", "first", or "uniform"
    normalize: str = "global"               # "global" or "per_frame"
    augment: bool = True

    @property
    def is_4d(self) -> bool:
        return self.input_mode == "4d"

    def describe(self) -> str:
        if not self.is_4d:
            return f"3D input, spatial target {tuple(self.target_shape)}"
        frames = "all" if self.num_frames is None else self.num_frames
        return (
            f"4D input, spatial target {tuple(self.target_shape)}, "
            f"frames={frames} ({self.frame_selection}/{self.frame_padding}), "
            f"normalize={self.normalize}"
        )


@dataclass(order=True)
class CandidateInfoTuple:
    """Class for keeping track subject/session info."""

    qu_motion_float: float
    file_path: str
    subject_str: str
    session_str: str
    run_int: int = None
    suffix_str: str = None
    augmentation_index: int = None
    sort_index: float = field(init=False, repr=False)

    def __hash__(self):
        return hash(self.file_path)

    @property
    def subject(self) -> str:
        return self.subject_str

    def __post_init__(self):
        # sort by qu_motion_float
        self.sort_index = self.qu_motion_float

    @property
    def path_to_file(self) -> str:
        return self.file_path


def get_subject(p):
    return os.path.split(os.path.split(os.path.split(p)[0])[0])[1][4:]


def get_session(p):
    return os.path.split(os.path.split(p)[0])[1][4:]


def get_uid(p):
    return f"{get_subject(p)}_{get_session(p)}"


def get_candidate_info_list(folder, df, candidates: List[str]):
    candidate_info_list = []
    df = df.reset_index()  # make sure indexes pair with number of rows

    for _, row in df.iterrows():
        candidate = row["subject_id"]
        if candidate in candidates:
            append_candidate(folder, candidate_info_list, row)

    candidate_info_list.sort(reverse=True)

    return candidate_info_list


def build_file_name(row):
    """Resolve the on-disk file name for a CSV row.

    Prefers the literal `scan` column when present. Field map files carry BIDS
    entities the old template could not express (e.g. `dir-AP`, `acq-`), so
    reconstructing the name from subject/session/run/suffix alone breaks on
    fmap inputs. Falls back to the original template for legacy CSVs.
    """
    if "scan" in row.index and isinstance(row["scan"], str) and row["scan"].strip():
        return row["scan"].strip()

    subject_str = row["subject_id"]
    session_str = row["session_id"]
    run_int = row["run_id"]
    suffix_str = row["suffix"]

    return f"{subject_str}_{session_str}_run-{run_int}_{suffix_str}.nii.gz"


def append_candidate(folder, candidate_info_list, row):
    subject_str = row["subject_id"]
    session_str = row["session_id"]
    run_int = row["run_id"]
    suffix_str = row["suffix"]
    file_name = build_file_name(row)
    file_path = os.path.join(folder, file_name)
    qu_motion_float = float(row["QU_motion"])
    candidate_info_list.append(
        CandidateInfoTuple(qu_motion_float, file_path, subject_str, session_str, run_int, suffix_str)
    )


def z_normalize(image, mask=None, per_frame=False):
    """Z-normalization (standardization) of image data.

    With `per_frame=True` each volume along the trailing (frame) axis of a 4D
    array is standardized independently. That removes global intensity offsets
    between frames -- which is usually NOT what you want for a motion QC target,
    since signal dropout between frames is itself evidence of motion. Default
    stays global.
    """
    if per_frame:
        if image.ndim != 4:
            raise ValueError(
                f"per_frame normalization requires a 4D array, got ndim={image.ndim}"
            )
        out = np.empty_like(image)
        for t in range(image.shape[-1]):
            out[..., t] = z_normalize(image[..., t], mask=mask, per_frame=False)
        return out

    if mask is not None:
        masked_data = image[mask > 0]
        mean = np.mean(masked_data)
        std = np.std(masked_data)
    else:
        mean = np.mean(image)
        std = np.std(image)

    if std == 0:
        return image - mean
    return (image - mean) / std


def resize_or_pad(image, target_shape=DEFAULT_3D_TARGET_SHAPE):
    """Center crop and/or zero-pad an array of arbitrary rank to target_shape.

    Generalized from the original 3D-only implementation. `target_shape` must
    have the same rank as `image`; for 4D inputs the frame axis is handled
    separately by `fit_frames` before this is called, so target_shape here is
    still the 3 spatial dimensions.
    """
    current_shape = image.shape

    if len(current_shape) != len(target_shape):
        raise ValueError(
            f"resize_or_pad rank mismatch: image has shape {current_shape} "
            f"but target_shape is {tuple(target_shape)}"
        )

    padded_image = np.zeros(target_shape, dtype=image.dtype)

    # calculate slice positions to center the image
    slices_in = []
    slices_out = []

    for i in range(len(target_shape)):
        if current_shape[i] <= target_shape[i]:
            # need to pad - take all of input, place in center of output
            start_out = (target_shape[i] - current_shape[i]) // 2
            slices_out.append(slice(start_out, start_out + current_shape[i]))
            slices_in.append(slice(None))
        else:
            # need to crop - take center of input, fill all of output
            start_in = (current_shape[i] - target_shape[i]) // 2
            slices_in.append(slice(start_in, start_in + target_shape[i]))
            slices_out.append(slice(None))

    padded_image[tuple(slices_out)] = image[tuple(slices_in)]

    return padded_image


def fit_frames(image, num_frames, selection="center", padding="edge"):
    """Force the trailing (frame) axis of a 4D array to exactly `num_frames`.

    Field map acquisitions carry only a handful of volumes and the count varies
    by scanner platform, so this normalizes it.

    selection:
        "center"  - center crop when there are too many frames
        "first"   - keep the leading frames
        "uniform" - evenly spaced subsample across the acquisition
    padding:
        "edge"    - repeat the final frame (keeps realistic image content)
        "zero"    - append blank volumes
    """
    if num_frames is None:
        return image

    if image.ndim != 4:
        raise ValueError(f"fit_frames requires a 4D array, got ndim={image.ndim}")

    current_frames = image.shape[-1]

    if current_frames == num_frames:
        return image

    if current_frames > num_frames:
        if selection == "first":
            idx = np.arange(num_frames)
        elif selection == "uniform":
            idx = np.linspace(0, current_frames - 1, num_frames).round().astype(int)
        else:  # center
            start = (current_frames - num_frames) // 2
            idx = np.arange(start, start + num_frames)
        return image[..., idx]

    # too few frames - pad out to num_frames
    deficit = num_frames - current_frames
    if padding == "zero":
        pad_block = np.zeros(image.shape[:-1] + (deficit,), dtype=image.dtype)
    else:  # edge
        pad_block = np.repeat(image[..., -1:], deficit, axis=-1)

    log.debug(
        f"Padding frame axis from {current_frames} to {num_frames} using '{padding}'"
    )
    return np.concatenate([image, pad_block], axis=-1)


def _apply_spatial(image, fn):
    """Apply a purely spatial function to a 3D array, or frame-wise to a 4D one.

    Critical for 4D: the same transform must hit every frame identically, and it
    must never operate across the frame axis. The original augmentations rotated
    over axis pairs (0,1), (0,2), (1,2) and shifted with a 3-element vector,
    which on a 4D array would smear signal through time.
    """
    if image.ndim == 3:
        return fn(image)

    out = [fn(image[..., t]) for t in range(image.shape[-1])]
    return np.stack(out, axis=-1)


def random_flip_lr(image, prob=0.5):
    """Random left-right flip (axis 0 assumed to be left-right)."""
    if np.random.random() < prob:
        return np.flip(image, axis=0)  # Assuming first axis is left-right
    return image


def random_affine_transform(image, prob=0.8):
    """Simple random affine transformation using scipy.

    Transform parameters are drawn once and then applied identically to every
    frame, so a 4D field map stays internally consistent after augmentation.
    """
    if np.random.random() > prob:
        return image

    # Small random rotation (in degrees)
    angle = np.random.uniform(-5, 5)

    # Small random translation (spatial axes only)
    translation = [np.random.uniform(-2, 2) for _ in range(3)]

    # Decide up front which axis pairs rotate, so all frames match
    rotate_axes = [
        axis for axis in [(0, 1), (0, 2), (1, 2)] if np.random.random() < 0.3
    ]

    def _transform(volume):
        for axis in rotate_axes:
            volume = ndimage.rotate(
                volume, angle, axes=axis, reshape=False, order=1
            )
        return ndimage.shift(volume, translation, order=1)

    return _apply_spatial(image, _transform)


def load_scan_array(scan_path, spec: ScanSpec, is_val_set_bool=True):
    """Load a NIfTI file and preprocess it into a numpy array.

    Returns a 3D array for spec.input_mode == "3d" and a 4D array shaped
    (X, Y, Z, T) for "4d". Shared by both training and inference so the two
    paths cannot drift apart.
    """
    nii_img = nib.load(scan_path)
    image_data = nii_img.get_fdata()
    image_data = np.array(image_data, dtype=np.float32)

    if spec.is_4d:
        if image_data.ndim == 3:
            # A 3D file in a 4D run: promote to a single-frame 4D volume rather
            # than failing, so mixed fmap/anat CSVs still load.
            log.warning(
                f"Expected 4D input but {scan_path} is 3D; treating as one frame."
            )
            image_data = image_data[..., np.newaxis]
        elif image_data.ndim > 4:
            # Some vendors write trailing singleton dims.
            image_data = np.squeeze(image_data)
            if image_data.ndim != 4:
                raise ValueError(
                    f"{scan_path} has unsupported shape {image_data.shape}"
                )

        image_data = fit_frames(
            image_data,
            spec.num_frames,
            selection=spec.frame_selection,
            padding=spec.frame_padding,
        )
        image_data = _apply_spatial(
            image_data, lambda v: resize_or_pad(v, target_shape=tuple(spec.target_shape))
        )
    else:
        if image_data.ndim != 3:
            raise ValueError(
                f"Expected 3D input for input_mode='3d' but {scan_path} has "
                f"shape {image_data.shape}. Pass --input-mode 4d for field maps."
            )
        image_data = resize_or_pad(image_data, target_shape=tuple(spec.target_shape))

    image_data = z_normalize(
        image_data, per_frame=(spec.is_4d and spec.normalize == "per_frame")
    )

    # Apply augmentations only for training
    if not is_val_set_bool and spec.augment:
        image_data = random_flip_lr(image_data)

        if np.random.random() < 0.8:
            image_data = random_affine_transform(image_data)

    return image_data


def to_model_tensor(image_data, spec: ScanSpec):
    """Convert a preprocessed array to the canonical model input layout.

    Both modes return a 4D per-sample tensor (T, X, Y, Z), which the DataLoader
    batches to (B, T, X, Y, Z):

      - 3D inputs give T=1, which is byte-for-byte what the old
        `.unsqueeze(0)` produced, so existing checkpoints stay compatible.
      - 4D inputs move the NIfTI trailing frame axis to the front.

    The model wrapper then decides whether T means "input channels" or "frames
    to encode separately and pool over".
    """
    array = np.ascontiguousarray(image_data)

    if spec.is_4d:
        # (X, Y, Z, T) -> (T, X, Y, Z)
        array = np.moveaxis(array, -1, 0)
        return torch.from_numpy(np.ascontiguousarray(array))

    return torch.from_numpy(array).unsqueeze(0)


class AutoQcMRIs:
    def __init__(self, candidate_info, is_val_set_bool, spec: ScanSpec = None):
        spec = spec or ScanSpec()
        scan_path = candidate_info.path_to_file

        image_data = load_scan_array(scan_path, spec, is_val_set_bool)

        self.mri_image_tensor = to_model_tensor(image_data, spec)
        self.subject_session_uid = candidate_info

    def get_raw_candidate(self):
        return self.mri_image_tensor


@functools.lru_cache(1, typed=True)
def get_auto_qc_mris(candidate_info, is_val_set_bool, spec: ScanSpec = None):
    return AutoQcMRIs(candidate_info, is_val_set_bool, spec)


@raw_cache.memoize(typed=True)
def get_mri_raw_candidate(subject_session_uid, is_val_set_bool, spec: ScanSpec = None):
    auto_qc_mris = get_auto_qc_mris(subject_session_uid, is_val_set_bool, spec)
    mri_image_tensor = auto_qc_mris.get_raw_candidate()

    return mri_image_tensor


class AutoQcDataset(Dataset):
    def __init__(
        self,
        folder,
        subjects: List[str],
        df,
        output_df,
        is_val_set_bool=None,
        subject=None,
        sortby_str="random",
        spec: ScanSpec = None,
    ):
        self.is_val_set_bool = is_val_set_bool
        self.spec = spec or ScanSpec()
        self.candidateInfo_list = copy.copy(
            get_candidate_info_list(folder, df, subjects)
        )

        if subject:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.subject_str == subject
            ]

        if sortby_str == "random":
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == "QU_motion":
            self.candidateInfo_list.sort(key=lambda x: x.qu_motion_float)
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        log.info(
            "{!r}: {} {} samples ({})".format(
                self,
                len(self.candidateInfo_list),
                "validation" if is_val_set_bool else "training",
                self.spec.describe(),
            )
        )
        if output_df is not None:
            for candidate_info in self.candidateInfo_list:
                row_location = (
                    df["subject_id"] == candidate_info.subject
                ) & (df["session_id"] == candidate_info.session_str)
                output_df.loc[row_location, "training"] = 0 if is_val_set_bool else 1
                output_df.loc[row_location, "validation"] = 1 if is_val_set_bool else 0

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidate_info = self.candidateInfo_list[ndx]
        candidate_a = get_mri_raw_candidate(
            candidate_info, self.is_val_set_bool, self.spec
        )
        candidate_t = candidate_a.to(torch.float32)

        qu_motion = candidate_info.qu_motion_float
        qu_motion_t = torch.tensor(qu_motion, dtype=torch.float32)

        return (
            candidate_t,
            qu_motion_t,
            candidate_info.subject_str,
            candidate_info.session_str,
        )
