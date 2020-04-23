# -*- coding: UTF-8 -*-

import argparse
import json
import os

import numpy as np
import pandas as pd
import tools
from sklearn.externals import joblib
from time import time


# 2D数据参数测试
# parser.add_argument('--feature_csv', help='feature csv file, will take the first row as feature', default='./example-2D/output/filter/feature_selected.csv')
# parser.add_argument('--model', help='model file, .joblib', default='./example-2D/output/learn/models/xgboost/model.joblib')
# parser.add_argument('--label_encoder', help='json', default='./example-2D/output/learn/models/xgboost/encoder.npy')
# parser.add_argument('--feature_scalar', help='json', default='./example-2D/output/learn/scalar.joblib')
# parser.add_argument('--output', help='output csv file', default='./example-2D/output/infer/predict.json')

# 3D数据参数测试
# parser.add_argument('--feature_csv', help='feature csv file, will take the first row as feature', default='./example-3D/output/filter/feature_selected.csv')
# parser.add_argument('--model', help='model file, .joblib', default='./example-3D/output/models/SVM/model.joblib')
# parser.add_argument('--label_encoder', help='json', default='./example-3D/output/models/SVM/encoder.npy')
# parser.add_argument('--feature_scalar', help='json', default='./example-3D/output/scalar.joblib')
# parser.add_argument('--output', help='output csv file', default='./example-3D/predict.json')


def main(df_path, model_path, output_path, encoder_path, scalar_path, target_path=None):
    data = pd.read_csv(df_path)
    image_names = data['image']
    target_df = None
    if target_path is not None:
        target_df = pd.read_csv(target_path)
        data, label = tools.prepare_feature_n_label(data, target_df)
    data = data[[x for x in data.columns if x not in tools.keywords]]
    data = tools.preprocessing(data)
    scalar = joblib.load(scalar_path)
    try:
        data[data.columns] = scalar.transform(data)
    except ValueError:
        raise ValueError("输入特征需要和训练模型时选择的特征一致")

    # load model
    t1 = time()
    clf = joblib.load(model_path)

    predict_results = clf.predict_proba(data)
    res = list(np.load(encoder_path))
    predict_results = pd.DataFrame(predict_results, columns=res)
    predict_results['dataset'] = pd.Series(image_names)
    num_label = []
    for idx, row in label.iterrows():
        num_label.append(res.index(row['label']))
    predict_results['label'] = pd.Series(num_label, name='label')
    predict_results.to_csv(os.path.join(output_path, 'predict_scores.csv'), index=None)
    if target_df is not None:
        testing_results = []
        image_names = image_names.tolist()
        label = label['label'].tolist()
        for idx, row in predict_results.iterrows():
            node = {"Image_Name": image_names[idx], "Ground_Truth": label[idx]}
            pred = []
            for c in res:
                pred.append([str(c), float(row[c])])
            node['Prediction'] = pred
            testing_results.append(node)
        testing_results = {'Testing_Results': testing_results, 'DL_Type': "classification/case", "Class_Names": label}
        with open(os.path.join(output_path, 'result.json'), 'w') as f:
            json.dump(testing_results, f, indent=4, ensure_ascii=False)
    print("testing costs: {}".format(time() - t1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # debug
    parser.add_argument('--feature_csv', help='feature csv file, will take the first row as feature',
                        default='./example-debug/output/feature_selected.csv')
    parser.add_argument('--target_csv', help='target_csv', default=None)
    parser.add_argument('--model', help='model file, .joblib', default='./example-debug/output/models/knn/model.joblib')
    parser.add_argument('--label_encoder', help='json', default='./example-debug/output/models/xgboost/encoder.npy')
    parser.add_argument('--feature_scalar', help='json', default='./example-debug/output/scalar.joblib')
    parser.add_argument('--output', help='output directory', default='./example-debug/output/infer/predict.json')
    args = parser.parse_args()
    main(args.feature_csv, args.model, args.output, args.label_encoder, args.feature_scalar, args.target_csv)
