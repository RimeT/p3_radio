# Infer Scholar Radiomics API
API for Radiomics Analysis part in AI Scholar product.

## Feature extraction
```shell
python extract.py --data_csv --output --lib --cpus
```
图像提供两种格式，dicom文件夹或nifti文件，mask只支持nifti格式。

- data_csv: 包括image和mask两列，分别表示image和mask的路径
    - image: dicom文件夹（若为nrrd或nii单一文件，则需要在）
    - mask：单一文件
    
- output: 特征的导出文件路径，为csv文件，会包含image，mask作为图像信息

    |image|mask|feature1|feature|...|
    | :------ | ------- | ------- | ------- | ------- |
    |/data/1/image|/data/1/mask/1.nii|0.1|0.3||
    |/data/2/image|/data/2/mask/2.nii|1.2|0.9||

- lib: 组学库，选RIA 或 Pyradiomics （大小写无关，默认Pyradiomics） [TODO]

- cpus 线程数（CPU核数）

- img_reader 对图像读取是选择序列读取器，或是单一文件读取器，序列需要为dicom，单一文件需要为nrrd格式或nii格式。可接受参数为dicom,nii,nrrd。

## Feature selection
```shell
python feature_filter.py --feature_csv --target_csv --filters  --output_dir
```

- feature_csv 特征csv文件，所有列均为特征

- target_csv 标签文件，只有mask和label列，label列为标签列

- filters: 过滤算法（有顺序的，逗号分隔的），如 variance, kbest, lasso

- output_dir: 输出结果的文件夹

  ​	输出结果包括：

  - selection_result.csv：1代表这个特征被某个算法选取，0代表没有；如果特征被某个算法没有选取，则之后进行的算法不会再选取该特征，即一个0出现之后，后面的算法皆为0

    |feature|variance|k-best|Lasso|
    | :------ | :------ | ------- | ------- |
    |feature1|1|0|0|
    |feature2|1|1|1|
    
  - feature_selected.csv 过滤后的特征，可以直接用于机器学习，有image和mask两列，之后的列均为特征列

    | image         | mask               | feature1 | feature | ...  |
    | :------------ | ------------------ | -------- | ------- | ---- |
  | /data/1/image | /data/1/mask/1.nii | 0.1      | 0.3     |      |
    | /data/2/image | /data/2/mask/2.nii | 1.2      | 0.9     |      |

    - variance.json variance算法的结果，用于显示哪类特征还剩多少
    
      ```json
      [
          {
              "after": 58, 
              "before": 100, 
              "name": "original"
          },
          {
              "after": 58, 
              "before": 100, 
              "name": "original"
          }
      ]
      ```
      
    - kbest_pvalues.json k-best算法产生的p-values（基于上一步过滤出的特征）
    
        ```json
        {
            "exponential_firstorder_Energy": 0.017811809148548935, 
            "exponential_firstorder_Kurtosis": 0.5464124823894174, 
            "exponential_firstorder_Maximum": 0.006719362412181009, 
            "exponential_firstorder_Range": 0.011021240986750143, 
            "exponential_firstorder_Skewness": 0.4906445829352213, 
            "exponential_firstorder_TotalEnergy": 0.017811809148548935, 
            "exponential_gldm_DependenceNonUniformity": 0.03169727302199769
    }
      ```
    
    - lasso_alpha.json (lasso的最佳alpha值，存储为-log(alpha))
    
      ```json
      {"alpha": 0.007}
      ```
    ```
    
    - lasso_path.json (lasso path线，纯数组型，第一维代表每个特征的lasso线，第二维代表某条线上的每一个点[x, y])
    
      ```json
      [
          {
                "name": "original_shape_VoxelVolume", 
                "path": [
                    [
                        1.037725590033117, 
                        0.0
                    ], 
                    [
                        1.0680286203361473, 
                        0.0
                    ]
                ]
          },
          {
                "name": "original_shape_VoxelVolume", 
                "path": [
                    [
                        1.037725590033117, 
                        0.0
                    ], 
                    [
                        1.0680286203361473, 
                        0.0
                    ]
                ]
          }
      ]
    ```
      ![avatar](./raw/lasso_path.png)
    
    - lasso_mse.json (lasso MSE损失线，纯数组型，第一维代表有多少交叉验证线，第二维代表某条线上的每一个点[x, y])
    
      ```json
      [
          {
              "name": "fold0", 
              "path": [
                  [
                      1.037725590033117, 
                      0.2823885099307314
                  ], 
                  [
                      1.0680286203361473, 
                      0.28885440054522454
                  ]
              ]
          },
          {
              "name": "fold1", 
              "path": [
                  [
                      1.037725590033117, 
                      0.2823885099307314
                  ], 
                  [
                      1.0680286203361473, 
                      0.28885440054522454
                  ]
              ]
          }
      ]
    
      ```
      ![avatar](./raw/lasso_mse.png)
      
    - 其他raw data：均包括feature和chosen列，用于表示哪个feature是否被选择，以及对于算法的输出值，如方差，p值，coefficient值
    
        - raw_filter_variance.csv
        - raw_filter_kbest.csv
        - raw_filter_lasso.csv

## Feature analysis

```shell
python analyze.py --feature_csv --target_csv --k --output_dir
```

- feature_csv：特征csv文件，所有列均为特征

- target_csv：标签文件，只有mask和label列，label列为标签列

- k：聚类个数，默认为4

- output_dir: 导出分析结果的目录
  
    - stat.csv 统计结果
    
        | name      | type | missing | max  | min  | mean    | std       |
        | :-------- | :--- | :------ | :--- | :--- | :------ | :-------- |
        | feature_1 | Real | 0     | 0.1  | 0.1  | 0.1     | 0.1       |
        | feature_2 | Real | 1     | 0.1  | 0.1  | 0.1     | 0.1       |
    
    - pca.csv PCA分析结果，pca列为横轴，percent列为纵轴，折线值为percent的累加值
    
        |pca|percent|
        | :------ | :------ |
        |pc1|0.15|
        |pc2|0.06|

    - corr.csv 相关性分析结果

        | |feature1|feature2|featur3|feature|
        | :------ | :------ | :------ | :------ | :------ |
        |feature1|0.1|0.1|0.1|0.1|0.1|
        |feature2|0.1|0.1|0.1|0.1|0.1|
        |feature3|0.1|0.1|0.1|0.1|0.1|
        |feature4|0.1|0.1|0.1|0.1|0.1|

    - Clustering
      
        - cluster_stat.json 聚类基本结果
          
            ```json
            {
              "cluster1": {
                "number": 50,
                "percent": 0.166,
                "avg_distance": 12.27
              },
              "cluster2": {
                "number": 50,
                "percent": 0.166,
                "avg_distance": 12.27
              }
            }
            ```
        
        - cluster_centroids.csv 每个cluster的中心点 （K-means centroids chart）
          
            |class|Feature1|Feature2|Feature3|Feature4|
            | :------ | :------ | :------ | :------ | :------ |
            |cluster1|0.3|2.8|0.1|1.3|
            |cluster2|0.1|0.1|0.1|0.1|
            |cluster3|0.1|0.1|0.1|0.1|
            
        - cluster_voi.csv VOI分到哪个类别（Class）的详细结果
        
            |class|mask|image|Feature1|Feature2|
            | :------ | :------ | :------ | :------ | :------ |
            |cluster1||2.8|0.1|1.3|
            |cluster0|0.1|0.1|0.1|0.1|
            |cluster3|0.1|0.1|0.1|0.1|
            
        - 其他raw data
        
            - raw_analyze_pca.csv PCA分析的输出，给出每一个主成分的值是多少

## Machine learning

```shell
python learn.py --feature_csv --target_csv --tags_csv --models --output_dir
```

- feature_csv：特征csv文件，除了image和mask列之外所有列均为特征

- target_csv：标签文件，只有mask和label列，label列为标签列

- tags_csv: 标注训练集或者测试集的tag，有mask和dataset列

    - mask：mask文件路径
    - dataset: 数据集，**0为训练验证集，1为测试集**

    |mask|dataset|
    |:------ |:------ |
    |mask路径 |	1|
    |mask路径 |	1|
    |mask路径 |	0|

- models: 选择的模型，由逗号分隔，无顺序，可接受的模型为：

    - svm
    - bayes
    - knn
    - logistic
    - decision_tree
    - random_forest
    - xgboost
    - deep
    
- output_dir: 导出分析结果的目录
    - 结果总表: output_dir/model_compare.csv
    
      |model |train|valid|
      | :------ | :------ | :------ |
      |LR|0.76|0.62|
      |SVM|0.78|0.66|
      |XGBoost|0.78|0.66|

    - 每个模型的结果，两表一图: 
    
        - 模型结果路径：output_dir/models/model1/...
        
        - 表1：敏感性，特异性等结果：model1/report.csv

          |class|recall| F1-score| support| precision|
          | :------ | :------ | :------ | :------ | :------ |
          |class_1|0.8|0.3|27|0.7|
        
        - 表2：每个病灶的分类结果：model1/samples_result.csv

          |label|p_predict| p_predicted| p_label0| p_label1| correct|
          | :------ | :------ | :------ | :------ | :------ | :------ |
          |1|0|0.8|0.8|0.2|false|
        
        - 图：ROC曲线（对每个分类）model1/roc.json，tpr为纵轴，fpr为横轴

          ```json
  {
              "class1":{
                  "fpr": [0, 0.1, 0.3, 0.7, 1.0],
                  "tpr": [1.0, 0.8, 0.6, 0.3, 0.0],
                  "auc": 0.82
              },
              "class2":{
                  "fpr": [0, 0.1, 0.3, 0.7, 1.0],
                  "tpr": [1.0, 0.8, 0.6, 0.3, 0.0],
                  "auc": 0.82
              }
          }
          ```
          
        - 模型文件：model1/model.joblib
        
        - 标签encode文件：model1/encoder.npy
        
        - 归一化文件: models/scalar.joblib
        
        - 临时文件夹: temp
    
- auto_opt 自动调参，网格或随机搜索，回选取最佳结果，比较耗时，默认不开启

## Inference

```shell
python infer.py --feature_csv --model --label_encoder --feature_scalar --output
```

- feature_csv： csv文件，默认取第一行进行预测，只包括feature列

- model： joblib类型的model文件

- label_encoder: 类别的encoder文件，用于将模型预测的数值类型转换为真实类别名称

- feature_scalar: 归一化文件

- output：输出类别和概率，例：/output/predict.json
    ```json
    {
      "Invasion": 0.77,
      "pre-Invasion": 0.23
    }
    ```

------
Updated on 2019-08-20