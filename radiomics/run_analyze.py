import argparse

import pandas as pd
import analyze

from scholar.schema import schema, radiomics as projects
from scholar.radiomics.data import load_feature_data, load_data_index

def before_run(project_id, data_csv, target_csv):
    project, = projects[schema.projectId == project_id]
    features = project >> 'data'

    index_df = load_data_index(features)

    # generate data frame
    intent_query = schema['intent'] == 'in'
    active_query = schema['active']
    type_query = schema.type != 'object'
    data_df = pd.concat([load_feature_data(feature)
                        for feature in features[intent_query & active_query & type_query]],
                        axis=1)
    data_df = pd.concat([index_df[['image', 'mask']], data_df], axis=1)
    data_df = data_df.loc[index_df['flag'] == 1]
    data_df.to_csv(data_csv, index=False, encoding='utf-8')

    # generate target frame
    target_df = pd.concat([load_feature_data(feature)
                          for feature in features[schema.intent == 'out']],
                          axis=1)
    target_df.columns = ['label']
    target_df = pd.concat([index_df[['image', 'mask']], target_df], axis=1)
    target_df = target_df.loc[index_df['flag'] == 1]
    target_df.to_csv(target_csv, index=False, encoding='utf-8')


def main(args):
    before_run(args.project_id, args.feature_csv, args.target_csv)
    analyze.main(args.feature_csv, args.target_csv, args.output_dir, args.k)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # debug
    parser.add_argument('--feature_csv', help='feature csv file', default='./example-debug/output/feature_selected.csv')
    parser.add_argument('--target_csv', help='target csv file', default='./example-debug/target.csv')
    parser.add_argument('--output_dir', help='output csv file', default='./example-debug/output')
    parser.add_argument('--k', help='number of clusters', type=int, default=4)
    parser.add_argument('--project_id', help='analyze task project id')
    args = parser.parse_args()

    main(args)
