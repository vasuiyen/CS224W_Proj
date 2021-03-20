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

Setting up DVC for pushing new results: 
```
dvc remote add cs224w_put s3://cs224w-final-project-experiments/results/
dvc remote modify cs224w_put region eu-central-1
dvc remote modify cs224w_put credentialpath ~/.aws/credentials
dvc remote modify cs224w_put profile <AWSProfile>
```

Setting up DVC for getting the results: 
```
dvc remote add cs224w_get https://cs224w-final-project-experiments.s3.eu-central-1.amazonaws.com/results/
dvc pull -r cs224w_get

```

Push a new folder from the main results folder: 
```
dvc add MyNewFolder
dvc push MyNewFolder.dvc -r cs224w_put
```

Get all the results: 
```
dvc pull -r cs224w_get
```



