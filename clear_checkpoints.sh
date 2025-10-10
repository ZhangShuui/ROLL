cd /project/hdtaccuracy/trains/gspo/qwen3_1_7b_choice_v4/output

for d in actor_train-*; do
  for ckpt in 50 100 150 250; do
    target="$d/checkpoint-$ckpt"
    if [ -d "$target" ]; then
      echo "Deleting $target"
      rm -rf "$target"
    fi
  done
done
