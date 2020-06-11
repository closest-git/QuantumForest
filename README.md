# QuantumForest

QuantumForest is a new lib on the model of differentiable decision trees. It has the advantages of both trees and neural networks. Experiments on large datasets show that QuantumForest has higher accuracy than both deep networks and best GBDT libs(XGBoost, Catboost, mGBDT,...). 

- Keeping **simple tree structure**，easy to use and explain the decision process 

- **Full differentiability** like neural networks. So we could train it with many powerful optimization algorithms (SGD, Adam, …), just like the training of deep CNN.

- Support **batch training** to reduce the memory usage greatly.

- Support the **end-to-end learning** mode. 
Reduce a lot of work on data preprocessing and feature engineering. 
  

![](Differentiable tree_1.png)

  

## Performance

To verify our model and algorithm, we test its performance on **six large datasets**. 

​													Table 1: Six large tabular datasets

|             |     Higgs      |     Click      |    YearPrediction    |  Microsoft   |        Yahoo        | EPSILON               |
| ----------- | :------------: | :------------: | :------------------: | :----------: | :-----------------: | :-------------------- |
| Training    |      8.4M      |      800K      |         309K         |     580K     |        473K         | 320K                  |
| Validation  |      2.1M      |      100K      |         103K         |     143      |         71K         | 80K                   |
| Test        |      500K      |      100K      |         103K         |     241K     |        165K         | 100K                  |
| Features    |       28       |       11       |          90          |     136      |         699         | 2000                  |
| Problem     | Classification | Classification |      Regression      |  Regression  |     Regression      | Classification        |
| Description |  UCI ML Higgs  |  2012 KDD Cup  | Million Song Dataset | MSLR-WEB 10k | Yahoo LETOR dataset | PASCAL Challenge 2008 |



The following table lists the accuracy of QuantumForest and some GBDT libraries

|               | Higgs      | Click      | YearPrediction | Microsoft  | Yahoo      | EPSILON    |
| ------------- | ---------- | ---------- | -------------- | ---------- | ---------- | ---------- |
| CatBoost      | 0.2434     | 0.3438     | 80.68          | 0.5587     | 0.5781     | 0.1119     |
| XGBoost       | 0.2600     | 0.3461     | 81.11          | 0.5637     | 0.5756     | 0.1144     |
| LightGBM      | **0.2291** | 0.3322     | 76.25          | 0.5587     | **0.5576** | 0.1160     |
| NODE          | 0.2412     | **0.3309** | 77.43          | 0.5584     | 0.5666     | **0.1043** |
| mGBDT         | OOM        | OOM        | 80.67          | OOM        | OOM        | OOM        |
| QuantumForest | 0.2467     | **0.3309** | **74.02**      | **0.5568** | 0.5656     | 0.1048     |

*Some results are copied form the testing results of NODE 

 All libraries use default parameters. LightGBM is the winner of ’Higgs’ and ’Yahoo’ datasets. NODE is the winner of ’Click’ datasets. QuantumForest performs best on the ’Click’, ’YearPrediction’,’Microsoft’, and ’EPSILON’ datasets. mGBDT always failed because out of memory(OOM) for most large datasets.  The differentiable forest model has only been developed for a few years and is still in its early stages. QuantumForest shows the potential of differentiable forest model. 



## Dependencies

- Python 3.7

- PyTorch 1.3.1

  

## Usage

#### For six large dataset in Table 1:

1. Create a directory ***data_root*** for storing datasets

   QuantumForest would automatically download the datasets and save the files at ***data_root*** . Then automatically split each dataset into training, validation and test sets.

2. To get the accuracy of QuantumForest 

   For example, test the accuracy of the *HIGGS* dataset

   ```
   python main_tabular_data.py --data_root=../Datasets/ --dataset=HIGGS --learning_rate=0.002
   ```

   To test other datasets, set the *dataset* to one of *[YEAR,YAHOO,CLICK,MICROSOFT,HIGGS,EPSILON]*

3. To get the accuracy of other GBDT libraries (all with default parameters)

   For example, test the accuracy of LightGBM

   ```python
   python main_tabular_data.py  --dataset=HIGGS --model=LightGBM 
   ```

   To test other GBDT libraries, set the  *model* to *LightGBM, XGBoost, or Catboost* 

#### For other datasets:

1 Prepare the **data**

​	The class of data should be the child class of quantum_forest.TabularDataset. For the detail, please 			see TabularDataset.py.

2 Call quantum_forest

```python
import quantum_forest

config = quantum_forest.QForest_config(data,0.002)     
config.device = quantum_forest.OnInitInstance(random_state=42)
config.model="QForest"      
config.in_features = data.X_train.shape[1]
config.tree_module = quantum_forest.DeTree 
config, visual =  quantum_forest.InitExperiment(config, 0)
config.response_dim = 3
config.feat_info = None
data.onFold(0,config,pkl_path=f"FOLD_0.pickle")
learner = quantum_forest.QuantumForest(config,data,feat_info=None,visual=visual).   \
                fit(data.X_train, data.y_train, eval_set=[(data.X_valid,data.y_valid)])
best_score = learner.best_score
```



## Notes

More parameters

```
python main_tabular_data.py  --dataset=HIGGS --learning_rate=0.001 --scale=large --subsample=0.3
```

1. set different *learning_rate*.
2. set *scale* to *large*, the accuracy will be higher, but it will take longer to run.
3. set *subsample*<1.0, the accuracy maybe higher with less time.

## Citation

If you find this code useful, please consider citing:

```
[1] Chen, Yingshi. "Deep differentiable forest with sparse attention for the tabular data." arXiv preprint arXiv:2003.00223 (2020).
[2] Chen, Yingshi. "LiteMORT: A memory efficient gradient boosting tree system on adaptive compact distributions." arXiv preprint arXiv:2001.09419 (2020).
```

## Future work

- More huge testing datasets.

  ​	If anyone has larger dataset, please send  to us for testing

- More models.	

- More papers.


## Author

Yingshi Chen (gsp.cys@gmail.com)

