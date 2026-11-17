"""Detect the dimensionality of a cohort of NIfTI files.

The network's spatial rank is fixed when the model is built, so dimensionality
is a property of the *dataset*, not of individual files. This module surveys
headers and reports one answer for the whole cohort, with enough detail to say
why when files disagree.

Header dimensions are not taken at face value. A file may declare dim0=4 while
holding a single volume (65 of the 4412 HBCD field maps do exactly that), and a
volume shaped (X, Y, 1) is a 2D image wearing a 3D header. What matters is the
*effective* rank: how many axes actually have extent greater than one.
"""

import os
from collections import Counter

import nibabel as nib

from util.logconf import logging

log = logging.getLogger(__name__)

VALID_MODES = ("2d", "3d", "4d")


def effective_dims(path):
    """Return (spatial_dims, n_frames) from a NIfTI header, reading no voxels.

    spatial_dims is 2 or 3 -- the number of leading axes with extent > 1.
    n_frames is the length of the 4th axis, or 1 when there is no 4th axis or
    it is degenerate.

    Raises OSError if the header cannot be read; callers decide whether a
    missing file is fatal.
    """
    header = nib.load(path).header
    dim = header["dim"]
    ndim = int(dim[0])
    if ndim < 1:
        raise ValueError(f"{path}: header declares dim0={ndim}")

    extents = [int(dim[i]) for i in range(1, min(ndim, 4) + 1)]
    spatial = extents[:3]
    n_frames = extents[3] if len(extents) >= 4 else 1
    n_frames = max(int(n_frames), 1)

    spatial_dims = sum(1 for s in spatial if s > 1)
    # A genuinely 1D image is not something this pipeline handles; treat it as
    # 2D so the caller's validation produces a comprehensible error.
    spatial_dims = max(spatial_dims, 2)

    return spatial_dims, n_frames


def mode_for(spatial_dims, n_frames):
    """Map a file's measured shape onto one of VALID_MODES."""
    if n_frames > 1:
        return "4d"
    return "2d" if spatial_dims == 2 else "3d"


def survey(folder, scans, limit=200):
    """Inspect up to `limit` scans and summarise what the cohort looks like.

    Args:
        folder: root the `scans` paths are relative to.
        scans: iterable of paths relative to `folder`.
        limit: how many files to read; 0 or None reads every one. Headers are
            cheap but a 4000-file survey over shared storage is not free, and
            a few hundred settles the question in practice.

    Returns:
        dict with `modes` (Counter), `frames` (Counter), `spatial` (Counter),
        `checked`, `unreadable` (list of (path, reason)), and `target_shape`
        (the most common spatial extent, useful as a default).
    """
    scans = list(scans)
    if limit:
        scans = scans[:limit]

    modes, frames, spatial, shapes = Counter(), Counter(), Counter(), Counter()
    unreadable = []

    for rel in scans:
        path = os.path.join(folder, str(rel))
        try:
            sd, nf = effective_dims(path)
        except Exception as exc:  # unreadable, missing, or malformed header
            unreadable.append((rel, str(exc)))
            continue
        modes[mode_for(sd, nf)] += 1
        frames[nf] += 1
        spatial[sd] += 1
        try:
            dim = nib.load(path).header["dim"]
            shapes[tuple(int(dim[i]) for i in (1, 2, 3))] += 1
        except Exception:
            pass

    return {
        "modes": modes,
        "frames": frames,
        "spatial": spatial,
        "checked": len(scans) - len(unreadable),
        "unreadable": unreadable,
        "target_shape": shapes.most_common(1)[0][0] if shapes else None,
    }


def describe(result):
    """One-line human summary of a survey result."""
    modes = ", ".join(f"{m}:{n}" for m, n in result["modes"].most_common())
    frames = ", ".join(f"{f} frame(s):{n}" for f, n in result["frames"].most_common(5))
    return (f"{result['checked']} files checked | modes [{modes}] | {frames}")


def resolve_input_mode(requested, folder, scans, limit=200, allow_mixed=False):
    """Decide the cohort's input mode, honouring an explicit request.

    `requested` is one of VALID_MODES or "auto".

    With "auto" the majority mode wins and a mixed cohort is an error unless
    `allow_mixed` is set, because the model is built for one spatial rank and
    one frame-handling strategy. With an explicit mode the survey still runs,
    but disagreement is a warning rather than an error -- forcing a mode is a
    legitimate thing to do, for example reading 4D files as 3D by taking one
    frame.

    Returns (mode, survey_result).
    """
    requested = (requested or "auto").lower()
    if requested not in VALID_MODES + ("auto",):
        raise ValueError(
            f"input_mode must be one of {', '.join(VALID_MODES)} or auto, "
            f"got {requested!r}"
        )

    result = survey(folder, scans, limit=limit)

    if result["checked"] == 0:
        if requested == "auto":
            raise ValueError(
                "input_mode=auto could not read any headers under "
                f"{folder}. First failures: {result['unreadable'][:3]}"
            )
        log.warning(
            f"input_mode={requested} forced; no headers were readable under "
            f"{folder}, so nothing was validated against the data."
        )
        return requested, result

    log.info(f"dimensionality survey: {describe(result)}")
    if result["unreadable"]:
        log.warning(
            f"{len(result['unreadable'])} file(s) could not be read during the "
            f"survey, e.g. {result['unreadable'][0]}"
        )

    detected, n_detected = result["modes"].most_common(1)[0]
    mixed = len(result["modes"]) > 1

    if requested == "auto":
        if mixed and not allow_mixed:
            raise ValueError(
                "input_mode=auto found a mixed cohort: "
                f"{dict(result['modes'])}. The network is built for one spatial "
                "rank and one frame-handling strategy, so pick a mode "
                "explicitly (INPUT_MODE=3d treats a 4D file as its selected "
                "frames) or set ALLOW_MIXED_DIMS=true to take the majority."
            )
        if mixed:
            log.warning(
                f"mixed cohort {dict(result['modes'])}; taking the majority "
                f"mode {detected!r} because ALLOW_MIXED_DIMS is set."
            )
        log.info(
            f"input_mode=auto resolved to {detected!r} "
            f"({n_detected}/{result['checked']} files)"
        )
        return detected, result

    if detected != requested:
        log.warning(
            f"input_mode={requested} was requested but the data looks like "
            f"{detected!r} ({dict(result['modes'])}). Proceeding as "
            f"{requested!r} -- make sure that is deliberate."
        )
    return requested, result
