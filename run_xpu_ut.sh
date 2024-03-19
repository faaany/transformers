excel_dir="$1"

echo "+++++++++remove excel dir if exists and create a new++++++++++++"
rm -fr $excel_dir 
mkdir $excel_dir


test_files=("test_cache_utils" "test_modeling_utils")
for file in "${test_files[@]}"
do 
	echo "+++++++++run test file $file++++++++++++++++"
	pytest tests/$file.py -rA -s -n auto --dist=loadfile -m "not (not_device_test or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_xla or require_torch_tensorrt_fx or require_apex or is_pt_flax_cross_test or is_pt_tf_cross_test or require_tf or require_flax or require_torch_neuroncore)" -k "not (tpu or cuda or npu or flax or tf)" --excelreport="${excel_dir}/${file}.xlsx" --make-reports="${file}" --timeout=600
	echo "+++++++++++++++++++++++done for $file+++++++++++++"
done 


test_folders=("benchmark" "deepspeed" "extended" "fsdp" "generation" "peft_integration" "quantization" "trainer" "pipelines")
for folder in "${test_folders[@]}"
do 
	echo "+++++++++run test folder $folder++++++++++++++++"
	pytest tests/$folder -rA -s -n auto --dist=loadfile -m "not (not_device_test or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_xla or require_torch_tensorrt_fx or require_apex or is_pt_flax_cross_test or is_pt_tf_cross_test or require_tf or require_flax or require_torch_neuroncore)" -k "not (tpu or cuda or npu or flax or tf)" --excelreport="${excel_dir}/${folder}.xlsx" --make-reports="${folder}" --timeout=600
	echo "+++++++++++++++++++++++done for $folder+++++++++++++"
done 


for x in {a..z}
do 
	echo "++++++++run models beginning with $x++++++++++++++++"
	pytest tests/models/${x}* -rA -s -n auto --dist=loadfile -m "not (not_device_test or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_xla or require_torch_tensorrt_fx or require_apex or is_pt_flax_cross_test or is_pt_tf_cross_test or require_tf or require_flax or require_torch_neuroncore)" -k "not (tpu or cuda or npu or flax or tf)" --excelreport="${excel_dir}/${x}_models.xlsx" --make-reports="${x}_models" --timeout=600
	echo "+++++++++++++++++++++++done for $x+++++++++++++"
done


echo "++++++++run models beginning with blip_2++++++++++++++++"
pytest tests/models/blip_2/ -rA -s -n auto --dist=loadfile -m "not (not_device_test or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_xla or require_torch_tensorrt_fx or require_apex or is_pt_flax_cross_test or is_pt_tf_cross_test or require_tf or require_flax or require_torch_neuroncore)" -k "not (tpu or cuda or npu or flax or tf)" --excelreport="${excel_dir}/blip_2_models.xlsx" --make-reports="blip_2_models" --timeout=600