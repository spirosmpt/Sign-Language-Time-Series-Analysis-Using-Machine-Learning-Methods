First create `data/` folder and place data following the structure bellow (`distances` folder will be autocreated) :

```bash
data
├── distances
│   ├── L2\\\\\\\_l
│   │   └── D.npy
│   ├── L2\\\\\\\_r
│   │   └── D.npy
│   ├── Proc-add\\\\\\\_l
│   │   └── D.npy
│   ├── Proc-add\\\\\\\_r
│   │   └── D.npy
│   ├── Proc\\\\\\\_l
│   │   └── D.npy
│   ├── Proc\\\\\\\_r
│   │   └── D.npy
│   ├── Quat\\\\\\\_l
│   │   └── D.npy
│   ├── Quat\\\\\\\_r
│   │   └── D.npy
│   ├── wtr\\\\\\\_l
│   │   └── D.npy
│   └── wtr\\\\\\\_r
│       └── D.npy
└── WLASL\\\\\\\_npy\\\\\\\_dataset\\\\\\\_100\\\\\\\_split
    ├── X\\\\\\\_test.npy
    ├── X\\\\\\\_train.npy
    ├── X\\\\\\\_val.npy
    ├── y\\\\\\\_test.npy
    ├── y\\\\\\\_train.npy
    └── y\\\\\\\_val.npy
```

## 1\. Calculating / Tuning Distances

#### 1\. Create independent distances i.e L2, Quat, Proc, wtr. These are saved to `data/distances` as shown bellow:

```bash
data/distances
├── L2\\\\\\\_l
│   └── D.npy
├── L2\\\\\\\_r
│   └── D.npy
├── Proc-add\\\\\\\_l
│   └── D.npy
├── Proc-add\\\\\\\_r
│   └── D.npy
├── Proc\\\\\\\_l
│   └── D.npy
├── Proc\\\\\\\_r
│   └── D.npy
├── Quat\\\\\\\_l
│   └── D.npy
├── Quat\\\\\\\_r
│   └── D.npy
├── wtr\\\\\\\_l
│   └── D.npy
└── wtr\\\\\\\_r
    └── D.npy
```

Select distance and run:

```bash
python src/distance\\\\\\\_tuning/pipeline.py
```

#### 2\. Grid search blending parameter in distance combination. Select two distances to be added together.

```bash
python src/distance\\\\\\\_tuning/grid\\\\\\\_search\\\\\\\_distance\\\\\\\_combinations.py
```

#### 3\. (Optional) Grid search to joint right and left hand distances.

```bash
python src/distance\\\\\\\_tuning/grid\\\\\\\_search\\\\\\\_joint\\\\\\\_hands\\\\\\\_frame.py
```



## 2\. SSL Train Embedding Network

To run for all combinations run the following command.

```bash 
bash run\\\\\\\_all\\\\\\\_experiments.sh
```



## 3\. Train Transformer with Embeddings

First define embedding method to be used

```bash
EMB\\\\\\\_METHOD="right\\\\\\\_quat"
common="--epochs 50 --batch\\\\\\\_size 32 --lr 3e-4 --wd 1e-4 --heads 8 --layers 4 --model\\\\\\\_dim 256 --use\\\\\\\_posenc"
```

1. Baselines
Run without hand encoder:

```bash
python src/training/train.py --data\\\\\\\_dir data/WLASL\\\\\\\_npy\\\\\\\_dataset\\\\\\\_100\\\\\\\_split  --no\\\\\\\_hand\\\\\\\_encoder --seed 1 $common

```

Run with hand encoder start from scratch:

```bash
python src/training/train.py --data\\\\\\\_dir data/WLASL\\\\\\\_npy\\\\\\\_dataset\\\\\\\_100\\\\\\\_split --seed 1 $common
```

Run with hand encoder start from scratch and freeze weights:

```bash
python src/training/train.py --data\\\\\\\_dir data/WLASL\\\\\\\_npy\\\\\\\_dataset\\\\\\\_100\\\\\\\_split --seed 1 --freeze\\\\\\\_encoder $common
```



2. Run augmenting with embeddings as a preprocessing step

```bash
echo "$EMB\\\\\\\_METHOD"
python src/training/train.py --data\\\\\\\_dir data/"$EMB\\\\\\\_METHOD"\\\\\\\_aug --epochs 50 --batch\\\\\\\_size 32 --lr 3e-4 --wd 1e-4 --seed 1337 --heads 8 --layers 4 --model\\\\\\\_dim 256 --use\\\\\\\_posenc
```

3. Run augmenting with embeddings inside network, should have similar results with 2.

```bash
echo "$EMB\\\\\\\_METHOD"
python src/training/train.py --data\\\\\\\_dir data/WLASL\\\\\\\_npy\\\\\\\_dataset\\\\\\\_100\\\\\\\_split --encoder\\\\\\\_ckpt ./outputs/"$EMB\\\\\\\_METHOD"\\\\\\\_triplets/dist\\\\\\\_encoder.pt --freeze\\\\\\\_encoder --seed 1 $common
```

4. Run augmenting with embeddings inside network and allow weights to be trained.

```bash
echo "$EMB\\\\\\\_METHOD"
python src/training/train.py --data\\\\\\\_dir data/WLASL\\\\\\\_npy\\\\\\\_dataset\\\\\\\_100\\\\\\\_split --epochs 50 --batch\\\\\\\_size 32 --lr 3e-4 --wd 1e-4 --seed 1337 --heads 8 --layers 4 --model\\\\\\\_dim 256 --use\\\\\\\_posenc --encoder\\\\\\\_ckpt ./outputs/"$EMB\\\\\\\_METHOD"\\\\\\\_triplets/dist\\\\\\\_encoder.pt
```

