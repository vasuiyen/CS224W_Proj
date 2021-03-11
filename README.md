# cs224w-final-project


## Training Instructions

Training on KarateClub:
```
python .\train.py --dataset pyg-karate --num_partitions 1
```
Note: We recommend KarateClub be used for debugging purpose only, since the graph is too small. 

Training on CORA: 
```
python .\train.py --dataset pyg-cora --num_partitions 1 --hidden_dim 128
```
