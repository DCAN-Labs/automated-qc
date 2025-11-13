#!/bin/bash 

set +x 
# determine bids directory, run folders, and run templates
tier1_data_dir="/scratch.global/lundq163/auto_qc_all_files"
skullstrip_data_dir="/scratch.global/lundq163/auto_qc_skullstripped_files"
code_dir="$(pwd)"

run_files_folder="${code_dir}/run_files.preproc_skullstrip_only" 
skullstrip_template="template.preproc_skullstrip_only"

# if processing run folders exist delete them and recreate
if [ -d "${run_files_folder}" ]; then
	rm -rf "${run_files_folder}"
	mkdir -p "${run_files_folder}/logs"
else
	mkdir -p "${run_files_folder}/logs"
fi

# counter to create run numbers
m=0
for i in `cd ${tier1_data_dir}; ls -d sub*`; do
	subj_id=`echo $i | awk  -F"-" '{print $2}' | awk -F"_" '{print $1}'`
	ses_id=`echo $i | awk  -F"-" '{print $3}' | awk -F"_" '{print $1}'`
	run_number=`echo $i | awk  -F"-" '{print $4}' | awk -F"_" '{print $1}'`
	suffix=`echo $i | awk -F"_" '{print $NF}' | awk -F"." '{print $1}'`
	sed -e "s|SUBJECTID|${subj_id}|g" -e "s|SESSIONID|${ses_id}|g" -e "s|RUNNUMBER|${run_number}|g" -e "s|SUFFIX|${suffix}|g" -e "s|TIER1DIR|${tier1_data_dir}|g" -e "s|OUTPUTDIR|${skullstrip_data_dir}|g" -e "s|CODEDIR|${code_dir}|g" ${code_dir}/${skullstrip_template} > ${run_files_folder}/run${m}
	m=$((m+1))
done

chmod +x -R ${run_files_folder}