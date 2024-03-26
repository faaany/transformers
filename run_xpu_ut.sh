excel_dir="$1"

# echo "+++++++++remove excel dir if exists and create a new++++++++++++"
rm -fr $excel_dir 
mkdir $excel_dir

# echo "+++++++++run test file++++++++++++++++"
pytest tests/*.py -rA -s -n auto --dist=loadfile -m "not (not_device_test or require_ray or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_neuroncore or require_torch_tensorrt_fx or require_pytorch_quantization or require_apex or require_tf or require_flax or require_torch_xla or torch_fx or require_detectron2 or require_flash_attn or flash_attn_test or require_bitsandbytes or require_quanto or require_auto_gptq or require_auto_awq or require_aqlm or require_natten)" -k "not (tpu or cuda or npu or flax or tf)" --excelreport="${excel_dir}/single_files.xlsx" --make-reports="single_files" --timeout=600
# echo "+++++++++++++++++++++++done for single_files+++++++++++++"

test_folders=("benchmark" "bettertransformer" "deepspeed" "extended" "fsdp" "generation" "peft_integration" "quantization" "trainer" "pipelines")
for folder in "${test_folders[@]}"
do 
	echo "+++++++++run test folder $folder++++++++++++++++"
	pytest tests/$folder -rA -s -m "not (not_device_test or require_ray or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_neuroncore or require_torch_tensorrt_fx or require_pytorch_quantization or require_apex or require_tf or require_flax or require_torch_xla or torch_fx or require_detectron2 or require_flash_attn or flash_attn_test or require_bitsandbytes or require_quanto or require_auto_gptq or require_auto_awq or require_aqlm or require_natten)" -k "not (tpu or cuda or npu or flax or tf)" --excelreport="${excel_dir}/${folder}.xlsx" --make-reports="${folder}" --timeout=600
	echo "+++++++++++++++++++++++done for $folder+++++++++++++"
done 


for x in {a..z}
do 
	echo "++++++++run models beginning with $x++++++++++++++++"
	pytest tests/models/${x}* -rA -s -n auto --dist=loadfile -m "not (not_device_test or require_ray or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_neuroncore or require_torch_tensorrt_fx or require_pytorch_quantization or require_apex or require_tf or require_flax or require_torch_xla or torch_fx or require_detectron2 or require_flash_attn or flash_attn_test or require_bitsandbytes or require_quanto or require_auto_gptq or require_auto_awq or require_aqlm or require_natten)" -k "not (tpu or cuda or npu or flax or tf)" --excelreport="${excel_dir}/${x}_models.xlsx" --make-reports="${x}_models" --timeout=600
	echo "+++++++++++++++++++++++done for $x+++++++++++++"
done


# echo "===========collect tests that are not CPU-only==========="
# pytest -m "not (not_device_test or require_ray)" tests --collectonly -q
# echo "===========collect tests that are not Platform-only==========="
# pytest -m "not (not_device_test or require_ray or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_neuroncore or require_torch_tensorrt_fx or require_pytorch_quantization or require_apex or require_bitsandbytes or require_flash_attn)" -k "not (tpu or cuda or npu)" tests --collectonly -q
# echo "===========collect tests that are not relevant==========="
# pytest -m "not (not_device_test or require_ray or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_neuroncore or require_torch_tensorrt_fx or require_pytorch_quantization or require_apex or require_tf or require_flax or require_torch_xla or require_bitsandbytes or require_flash_attn)" -k "not (tpu or cuda or npu or flax or tf)" tests --collectonly -q
# echo "===========collect tests that are not supported yet==========="
# pytest -m "not (not_device_test or require_ray or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_neuroncore or require_torch_tensorrt_fx or require_pytorch_quantization or require_apex or require_tf or require_flax or require_torch_xla or torch_fx or require_bitsandbytes or require_flash_attn)" -k "not (tpu or cuda or npu or flax or tf)" tests --collectonly -q
# echo "===========collect tests that are not supported yet, but will support==========="
# pytest -m "not (not_device_test or require_ray or require_torch_up_to_2_gpus or require_torch_multi_gpu or require_torch_gpu or require_torch_non_multi_gpu or require_torch_bf16_gpu or require_torch_tf32 or require_torch_npu or require_torch_multi_npu or require_torch_neuroncore or require_torch_tensorrt_fx or require_pytorch_quantization or require_apex or require_tf or require_flax or require_torch_xla or torch_fx or require_detectron2 or flash_attn_test or require_quanto or require_auto_gptq or require_auto_awq or require_aqlm or require_natten or require_bitsandbytes or require_flash_attn)" -k "not (tpu or cuda or npu or flax or tf)" tests --collectonly -q