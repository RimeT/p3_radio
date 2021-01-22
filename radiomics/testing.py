# -*- coding: UTF-8 -*-

import argparse
import json
import os

import numpy as np
import pandas as pd
import tools
from sklearn.externals import joblib
from time import time


def _check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def save_result(predict_results, target_df, res, label, output_path, image_names, mask_names):
    predict_results['dataset'] = pd.Series(image_names)
    predict_results['mask'] = pd.Series(mask_names)
    if target_df is not None:
        num_label = []
        label_names = []
        for idx, row in label.iterrows():
            num_label.append(res.index(row['label']))
            label_names.append(row['label'])
        predict_results['label'] = pd.Series(num_label, name='label')
        predict_results['label_name'] = pd.Series(label_names, name='label_name')
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


def main(df_path, model_path, output_path, encoder_path, scalar_path, target_path, cv=0):
    # load test data
    data = pd.read_csv(df_path)
    image_names = data['image']
    mask_names = data['mask']
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
    res = list(np.load(encoder_path))
    if cv <= 1 and os.path.isfile(model_path):
        clf = joblib.load(model_path)
        predict_results = clf.predict_proba(data)
        predict_results = pd.DataFrame(predict_results, columns=res)
        save_result(predict_results, target_df, res, label, output_path, image_names, mask_names)
    else:
        for i in range(cv):
            fold_model = os.path.join(model_path, f'fold_{i}', 'model.joblib')
            out_sub_dir = _check_dir(os.path.join(output_path, f"fold_{i}"))
            clf = joblib.load(fold_model)
            predict_results = clf.predict_proba(data)
            predict_results = pd.DataFrame(predict_results, columns=res)
            save_result(predict_results, target_df, res, label, out_sub_dir, image_names, mask_names)
    print("testing costs: {}".format(time() - t1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # debug
    parser.add_argument('--feature_csv', help='feature csv file, will take the first row as feature',
                        default='./example-debug/output/feature_selected.csv')
    parser.add_argument('--target_csv', help='target_csv', default=None)
    parser.add_argument('--model', help='model file, .joblib')
    parser.add_argument('--label_encoder', help='json')
    parser.add_argument('--feature_scalar', help='json')
    parser.add_argument('--cv', default=0, type=int)
    parser.add_argument('--output', help='output directory')
    args = parser.parse_args()
    main(args.feature_csv, args.model, args.output, args.label_encoder, args.feature_scalar, args.target_csv, args.cv)
