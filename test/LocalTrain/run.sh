python run.py --partition noniid-label-distribution --beta 0.3 --yamlfile ../../experiments/DBPedia_LSTM_cross_device.yaml --use_wandb True --device cuda:0 &
python run.py --partition noniid-label-distribution --beta 1.0 --yamlfile ../../experiments/DBPedia_LSTM_cross_device.yaml --use_wandb True --device cuda:1 &
python run.py --partition shards --num_shards_per_client 3 --yamlfile ../../experiments/DBPedia_LSTM_cross_device.yaml --use_wandb True --device cuda:2 &
python run.py --partition shards --num_shards_per_client 5 --yamlfile ../../experiments/DBPedia_LSTM_cross_device.yaml --use_wandb True --device cuda:3 &
