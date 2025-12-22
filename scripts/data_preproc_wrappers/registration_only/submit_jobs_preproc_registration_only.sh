#!/bin/bash -l

array=$1

if [ -z "$array" ]

then
      echo ""
      echo "No job array specified! Please enter jobs to run as argument:"
      echo ""
      echo "    EXAMPLE 1:  submit_ALL.sh 0-99"
      echo "    EXAMPLE 2:  submit_ALL.sh 1-3,6,9"
      echo ""

else
      echo ""
      echo "Checking for output_logs folder..."
      if [ -d "$(pwd)/output_logs" ]; then
	      rm -rf "$(pwd)/output_logs"
	      mkdir -p "$(pwd)/output_logs"
            else
	mkdir -p "$(pwd)/output_logs"
            fi
      echo ""
      echo "output_logs folder found/created."
      
      echo ""
      echo "Submitting the following jobs for preproc now: $array"
      echo ""

      abcd=$(sbatch --parsable -a ${array} resources_preproc_registration_only.sh)

      echo "preproc JOB ID: $abcd"

      echo ""
      echo "Output logs will appear in output_logs folder. Use 'squeue -al --me' to monitor jobs."
      echo ""
fi

