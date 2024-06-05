export RUN_SLOW=1
export RUN_PT_TF_CROSS_TESTS="False"
export RUN_PT_FLAX_CROSS_TESTS="False"
export TRANSFORMERS_TEST_DEVICE="xpu"
export TRANSFORMERS_TEST_DEVICE_SPEC="spec.py"

excel_dir="$1"

echo "+++++++++remove excel dir if exists and create a new++++++++++++"
rm -fr $excel_dir 
mkdir $excel_dir

NOT_RUN_MARKERS="not (not_device_test or torch_fx or flash_attn_test)"
NOT_RUN_KEYWORDS="not (tpu or npu or cuda or flax or tf or ModelOnTheFlyConversionTester)"

start_time=$(date +%s)
echo "+++++++++run single test files++++++++++++++++"
pytest tests/*.py -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --excelreport="${excel_dir}/single_files.xlsx" --make-reports="single_files" --timeout=600
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "+++++++++++++++++++++++done for single_files; time(s): $elapsed+++++++++++++"

start_time_epoch=$(date +%s)
test_folders=("benchmark" "extended" "fsdp" "generation" "peft_integration" "quantization" "trainer" "pipelines")
for folder in "${test_folders[@]}"
do 
	echo "+++++++++run test folder $folder++++++++++++++++"
	pytest tests/$folder -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --excelreport="${excel_dir}/${folder}.xlsx" --make-reports="${folder}" --timeout=600
	echo "+++++++++++++++++++++++done for $folder+++++++++++++"
done 
end_time_epoch=$(date +%s)
elapsed=$(( end_time_epoch - start_time_epoch ))
echo "+++++++++++++++++++++++all test folders done, time(s): $elapsed+++++++++++++"

start_time_epoch=$(date +%s)
for x in {a..z}
do 
	echo "++++++++run models beginning with $x++++++++++++++++"
	pytest tests/models/${x}* -m "${NOT_RUN_MARKERS}" -k "${NOT_RUN_KEYWORDS}" --ignore tests/sagemaker --ignore tests/bettertransformer --excelreport="${excel_dir}/${x}_models.xlsx" --make-reports="${x}_models" --timeout=600
	echo "+++++++++++++++++++++++done for $x+++++++++++++"
done
end_time_epoch=$(date +%s)
elapsed=$(( end_time_epoch - start_time_epoch ))
echo "+++++++++++++++++++++++all models done, time(s): $elapsed+++++++++++++"