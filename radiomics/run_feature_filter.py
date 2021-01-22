import os
import argparse
import pandas as pd
import numpy as np

import feature_filter
from scholar.schema import schema, radiomics as projects
from scholar.radiomics.data import load_feature_data, load_data_index


def before_run(project_id, data_csv, target_csv):
    project, = projects[schema.projectId == project_id]
    features = project >> 'data'

    index_df = load_data_index(features)

    # generate data frame
    intent_query = schema.intent == 'in'
    type_query = schema.type != 'object'
    data_df = pd.concat([load_feature_data(feature)
                        for feature in features[intent_query & type_query]],
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

def after_run(filters, output_dir, project_id):
    # remove un-specified filter columns from result file
    result_df = pd.read_csv(os.path.join(output_dir, 'selection_result.csv'), index_col=False)
    for column in result_df.columns:
        if column == 'feature':
            continue
        if column == 'k-best' and 'k_best' in filters:
            continue
        if column == 'variance' and 'variance_threshold' in filters:
            continue
        if column == 'lasso' and 'lasso' in filters:
            continue
        result_df = result_df.drop(column, axis=1)
    result_df.to_csv(os.path.join(output_dir, 'selection_result.csv'), index=False, encoding='utf-8')

    result_df = result_df.set_index('feature')

    # update feature properties
    project, = projects[schema.projectId == project_id]

    for item in project.selectionParameters['filters']:
        if item['filter'] == 'k_best':
            item['resultCount'] = np.sum(result_df['k-best'])
        elif item['filter'] == 'variance_threshold':
            item['resultCount'] = np.sum(result_df['variance'])
        elif item['filter'] == 'lasso':
            item['resultCount'] = np.sum(result_df['lasso'])

        features = project >> 'data'
        for feature in features.nodes():
            try:
                row = result_df.loc[feature.name]
                feature['active'] = all(map(bool, row))
            except KeyError:
                feature['active'] = False

def main(args):
    filters = args.filters.split(',')
    before_run(args.project_id, args.feature_csv, args.target_csv)
    feature_filter.main(args.feature_csv, args.target_csv, args.output_dir, filters)
    after_run(filters, args.output_dir, args.project_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_csv', help='feature csv file', default='./example-2D/output/feature_ria.csv')
    parser.add_argument('--target_csv', help='target csv file', default='./example-2D/target.csv')
    parser.add_argument('--filters', help='filters', default='variance, k-best, lasso')
    parser.add_argument('--output_dir', help='output dir', default='./example-2D/output/filter')
    parser.add_argument('--project_id', help='selection task project id')
    args = parser.parse_args()
    main(args)
 