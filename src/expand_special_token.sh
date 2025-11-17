python -m src.expand_special_token \
  --collection_path /scratch/yx3044/Projects/improving-learned-index/required_files/collections/msmarco-passage/queries.train.tsv \
  --collection_type msmarco \
  --num_special_tokens 80 \
  --output_path /scratch/yx3044/Projects/improving-learned-index/required_files/collections/msmarco-passage/queries.train.exp80.tsv




python -m src.expand_special_token \
  --collection_path /scratch/yx3044/Projects/improving-learned-index/required_files/collections/msmarco-passage/collection.tsv \
  --collection_type msmarco \
  --num_special_tokens 80 \
  --output_path /scratch/yx3044/Projects/improving-learned-index/required_files/collections/msmarco-passage/collection.exp80.tsv 