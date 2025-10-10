python $(python -c "import transformers, pathlib; \
print(pathlib.Path(transformers.__file__).parent/'models'/'llama'/'convert_llama_weights_to_hf.py')") \
  --input_dir /path/to/meta_llama_raw \
  --model_size 1B \
  --llama_version 3.2 \
  --output_dir /path/to/output_hf
