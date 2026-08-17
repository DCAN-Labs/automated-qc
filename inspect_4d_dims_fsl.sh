#!/bin/bash
# Survey NIfTI dimensions with FSL before choosing --target-shape / --num-frames.
#
# Equivalent to scripts/utils/inspect_4d_dims.py, but uses fslhd so it needs no
# Python environment. Reads headers only, so it is fast over large cohorts.
#
# Usage:
#   module load fsl          # run `module avail fsl` to see versions on MSI
#   ./inspect_4d_dims_fsl.sh /scratch.global/$USER/auto_qc/fmaps '*_epi.nii.gz'
#
# The two NIfTI header fields that matter:
#   dim0 = how many dimensions the file declares (3 or 4)
#   dim4 = how many volumes/frames are in it
# These disagree more often than you would expect: a file can declare dim0=4
# while carrying dim4=1, which is a single volume wearing a 4D header. Those are
# the ones that will quietly get edge-padded up to --num-frames, so they are
# listed separately at the end.

set -u

DIRECTORY="${1:-}"
PATTERN="${2:-*.nii.gz}"

if [ -z "$DIRECTORY" ]; then
    echo "Usage: $0 <directory> [pattern]" >&2
    exit 1
fi

if ! command -v fslhd > /dev/null 2>&1; then
    echo "ERROR: fslhd not found. Run 'module load fsl' first." >&2
    exit 1
fi

if [ ! -d "$DIRECTORY" ]; then
    echo "ERROR: directory not found: $DIRECTORY" >&2
    exit 1
fi

TMP=$(mktemp) || exit 1
trap 'rm -f "$TMP"' EXIT

COUNT=0
for f in "$DIRECTORY"/$PATTERN; do
    [ -e "$f" ] || continue
    COUNT=$((COUNT + 1))
    fslhd "$f" | awk -v name="$f" '
        $1 == "dim0" {d0 = $2}
        $1 == "dim1" {d1 = $2}
        $1 == "dim2" {d2 = $2}
        $1 == "dim3" {d3 = $2}
        $1 == "dim4" {d4 = $2}
        END {print d0, d1, d2, d3, d4, name}
    ' >> "$TMP"
done

if [ "$COUNT" -eq 0 ]; then
    echo "No files matched $DIRECTORY/$PATTERN"
    exit 0
fi

echo "Scanned $COUNT files in $DIRECTORY matching '$PATTERN'"

echo
echo "=== shape histogram (count | dim0 dim1 dim2 dim3 dim4) ==="
awk '{print $1, $2, $3, $4, $5}' "$TMP" | sort | uniq -c | sort -rn

echo
echo "=== frame counts (dim4) ==="
awk '{print $5}' "$TMP" | sort -n | uniq -c

echo
echo "=== files that are not genuinely 4D ==="
NOT4D=$(awk '$1 < 4 || $5 < 2 {print "  " $6 "  dim0=" $1 " dim4=" $5}' "$TMP")
if [ -z "$NOT4D" ]; then
    echo "  (none)"
else
    echo "$NOT4D"
fi

echo
echo "=== suggested flags ==="
awk '
    function roundup8(v) { return int((v + 7) / 8) * 8 }
    {
        if ($2 > x) x = $2
        if ($3 > y) y = $3
        if ($4 > z) z = $4
        frames[NR] = $5
        n = NR
    }
    END {
        # median frame count
        for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++)
            if (frames[j] < frames[i]) { t = frames[i]; frames[i] = frames[j]; frames[j] = t }
        med = (n % 2) ? frames[(n + 1) / 2] : int((frames[n / 2] + frames[n / 2 + 1]) / 2)

        tx = roundup8(x); ty = roundup8(y); tz = roundup8(z)
        printf "  --target-shape %d,%d,%d\n", tx, ty, tz
        printf "  --num-frames %d\n", med

        # How many stride-2 stages the smallest target dim can absorb
        smallest = tx; if (ty < smallest) smallest = ty; if (tz < smallest) smallest = tz
        stages = 0; s = smallest
        while (int(s / 2) >= 2) { s = int(s / 2); stages++ }
        printf "\n  Smallest target dim is %d, supporting about %d stride-2 stages.\n", smallest, stages
        printf "  Default --net-strides is 2,2,2,2,2 (5 stages) against a 260x320x320\n"
        printf "  anatomical; trim it if the number above is smaller.\n"
    }
' "$TMP"

UNIQUE_FRAMES=$(awk '{print $5}' "$TMP" | sort -n | uniq | wc -l)
if [ "$UNIQUE_FRAMES" -gt 1 ]; then
    echo
    echo "  NOTE: frame count varies across the cohort ($UNIQUE_FRAMES distinct values),"
    echo "  so --frame-mode pool is the safer choice than channels."
fi
