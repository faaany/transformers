excel_dir="$1"

echo "+++++++++remove excel dir if exists and create a new++++++++++++"
rm -fr $excel_dir 
mkdir $excel_dir
 
echo "+++++++++run test file++++++++++++++++"
pytest tests/*.py -rA -s -n auto --dist=loadfile -m "not (not_device_test or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_xla or require_torch_tensorrt_fx or require_apex or is_pt_flax_cross_test or is_pt_tf_cross_test or require_tf or require_flax or require_torch_neuroncore or require_natten)" -k "not (tpu or cuda or npu or flax or tf)" --excelreport="${excel_dir}/single_files.xlsx" --make-reports="${file}" --timeout=600
echo "+++++++++++++++++++++++done for $file+++++++++++++"

test_folders=("benchmark" "bettertransformer" "deepspeed" "extended" "fsdp" "generation" "peft_integration" "quantization" "trainer" "pipelines")
for folder in "${test_folders[@]}"
do 
	echo "+++++++++run test folder $folder++++++++++++++++"
	pytest tests/$folder -rA -s -n auto --dist=loadfile -m "not (not_device_test or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_xla or require_torch_tensorrt_fx or require_apex or is_pt_flax_cross_test or is_pt_tf_cross_test or require_tf or require_flax or require_torch_neuroncore or require_natten)" -k "not (tpu or cuda or npu or flax or tf)" --excelreport="${excel_dir}/${folder}.xlsx" --make-reports="${folder}" --timeout=600
	echo "+++++++++++++++++++++++done for $folder+++++++++++++"
done 

mv models tests/

for x in {a..z}
do 
	echo "++++++++run models beginning with $x++++++++++++++++"
	pytest tests/models/${x}* -rA -s -n auto --dist=loadfile -m "not (not_device_test or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_xla or require_torch_tensorrt_fx or require_apex or is_pt_flax_cross_test or is_pt_tf_cross_test or require_tf or require_flax or require_torch_neuroncore or require_natten)" -k "not (tpu or cuda or npu or flax or tf)" --excelreport="${excel_dir}/${x}_models.xlsx" --make-reports="${x}_models" --timeout=600
	echo "+++++++++++++++++++++++done for $x+++++++++++++"
done