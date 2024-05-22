# Relation-enhanced Negative Sampling for Multimodal Knowledge Graph Completion

This is the code of the paper **Relation-enhanced Negative Sampling for Multimodal Knowledge Graph Completion** for ACM MM 2022.

## Requirements

- pytorch == 1.10.1
- numpy == 1.20.3



## Datasets

The source images and triples of MMKB-DB15K are from [mmkb](https://github.com/mniepert/mmkb). 

The source text of three datasets are from [DBpedia](https://www.dbpedia.org/).

The embeddings and raw data can be downloaded in the [Google Drive](https://drive.google.com/drive/folders/1sFC-P9RKnikqNXjmLcj0IX7x5zvRs-Yj?usp=drive_link)

## Usage

```
mkdir data models results
```

put the datasets in `./data` and 

```
python run_gumbel.py --do_train --do_valid --do_test --data_path=data/MMKB-DB15K --model=TransE -n=20 -d=200 -g=6 -a=0.5 \
        -r=0.0 -lr=0.0001 -kca_lr=0.0001 --sample_method=gumbel  --pre_sample_num=1500  --loss_rate=100 --exploration_temp=10 \
        --gpu=0  --max_steps=100000 --valid_steps=10000 -b=400
```


This code refers to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) and [Nscaching](https://github.com/yzhangee/NSCaching). 



## Citation
If you find this codebase useful in your research, please cite the following paper.

```
@inproceedings{xu2022relation,
  title={Relation-enhanced Negative Sampling for Multimodal Knowledge Graph Completion},
  author={Xu, Derong and Xu, Tong and Wu, Shiwei and Zhou, Jingbo and Chen, Enhong},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={3857--3866},
  year={2022}
}
```
