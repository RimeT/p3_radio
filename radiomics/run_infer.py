import os
import argparse
import json
import shutil
import infer

import pandas as pd
from scholar.schema import schema, radiomics as projects
from scholar.radiomics.data import load_feature_data, load_data_index


def before_run(project_id, row_id, feature_csv):
    project, = projects[schema.projectId == project_id]
    features = project >> 'data'

    index_df = load_data_index(features)

    # generate data frame
    intent_query = schema['intent'] == 'in'
    active_query = schema['active']
    data_df = pd.concat([load_feature_data(feature)
                        for feature in features[intent_query & active_query]],
                        axis=1)
    data_df = pd.concat([index_df[['image', 'mask']], data_df], axis=1)
    data_df = data_df.loc[index_df['id'] == str(row_id)]
    data_df.to_csv(feature_csv, index=False, encoding='utf-8')

    target_feature, = features[schema['intent'] == 'out']
    target_df = load_feature_data(target_feature)
    gt_label = target_df.loc[index_df['id'] == str(row_id)].values[0][0]
    return gt_label


def after_run(feature_csv, output, gt_label, job_dir):
    data_df = pd.read_csv(feature_csv, index_col=False)
    mask_path = data_df.iloc[0]['mask']
    image_path = data_df.iloc[0]['image']
    # write regionId to predict.json
    with open(output) as f:
        d = json.load(f)
        d['regionId'] = os.path.basename(mask_path.split('.')[0])
        d['gtLabel'] = gt_label
    with open(output, 'w') as f:
        json.dump(d, f)

    # copy image data and regionfile
    shutil.copytree(image_path,
                    os.path.join(job_dir, os.path.basename(image_path)))
    shutil.copy(os.path.join(os.path.dirname(image_path), 'images.lst'),
                os.path.join(job_dir, 'images.lst'))
    shutil.copy(os.path.join(os.path.dirname(image_path), 'regionfile'),
                os.path.join(job_dir, 'regionfile'))
    shutil.copy(os.path.join(os.path.dirname(image_path), '_data.json'),
                os.path.join(job_dir, 'item.json'))


def main(args):
    gt_label = before_run(args.project_id, args.row_id, args.feature_csv)
    infer.main(args.feature_csv, args.model, args.output, args.label_encoder, args.feature_scalar)
    after_run(args.feature_csv, args.output, gt_label, args.job_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # debug
    parser.add_argument('--feature_csv', help='feature csv file, will take the first row as feature', default='./example-debug/output/feature_selected.csv')
    parser.add_argument('--model', help='model file, .joblib', default='./example-debug/output/models/knn/model.joblib')
    parser.add_argument('--label_encoder', help='json', default='./example-debug/output/models/xgboost/encoder.npy')
    parser.add_argument('--feature_scalar', help='json', default='./example-debug/output/scalar.joblib')
    parser.add_argument('--output', help='output csv file', default='./example-debug/output/infer/predict.json')
    parser.add_argument('--project_id', help='infer task project id')
    parser.add_argument('--row_id', help='infer task row id')
    parser.add_argument('--job_dir', help='infer task job dir')
    args = parser.parse_args()
    main(args)
