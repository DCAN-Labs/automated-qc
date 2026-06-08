module load ants

# Input arguments
MRI_IN=$1       # input MRI image
MRI_OUT=$2      # output registered MRI image
REF=$3          # reference template image
REG=$4          # output prefix for registration transforms

# Construct the antsRegistration command
cmd=$(cat <<EOF
antsRegistration --collapse-output-transforms 1 \
  --dimensionality 3 \
  --float 1 \
  --initialize-transforms-per-stage 0 \
  --interpolation LanczosWindowedSinc \
  --output [${REG}, ${MRI_OUT}] \
  --transform Rigid[0.05] \
  --metric Mattes[${REF}, ${MRI_IN}, 1, 56, Regular, 0.25] \
  --convergence [100x100, 1e-06, 20] \
  --smoothing-sigmas 2.0x1.0vox \
  --shrink-factors 2x1 \
  --use-histogram-matching 1 \
  --transform Affine[0.08] \
  --metric Mattes[${REF}, ${MRI_IN}, 1, 56, Regular, 0.25] \
  --convergence [100x100, 1e-06, 20] \
  --smoothing-sigmas 1.0x0.0vox \
  --shrink-factors 2x1 \
  --use-histogram-matching 1 \
  --transform SyN[0.1, 3.0, 0.0] \
  --metric CC[${REF}, ${MRI_IN}, 1, 4, None, 1] \
  --convergence [100x70x50x20, 1e-06, 10] \
  --smoothing-sigmas 3.0x2.0x1.0x0.0vox \
  --shrink-factors 8x4x2x1 \
  --use-histogram-matching 1 \
  --winsorize-image-intensities [0.005, 0.995] \
  --write-composite-transform 1 \
  -v
EOF
)

# Print and execute the command
echo "Running command:"
echo "${cmd}"
eval "${cmd}"

# Check the command's success
if [ $? -eq 0 ]; then
  echo "Registration completed successfully."
else
  echo "Registration failed. Please check the inputs and try again."
  exit 1
fi