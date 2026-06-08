module load ants

# Input arguments
MRI_IN=$1       # input MRI image
MRI_OUT=$2      # output registered MRI image
REF=$3          # reference template image
REG=$4          # output prefix for registration transforms

# Construct the antsRegistration command for rigid alignment only
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