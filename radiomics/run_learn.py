import argparse
import pandas as pd
import learn

from scholar.schema import schema, radiomics as projects
from scholar.radiomics.data import load_feature_data, load_data_index

def before_run(project_id, data_csv, target_csv, tags_csv):
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

    # generate tags
    tags_df = index_df[['image', 'mask', 'tag']][index_df['flag'] == 1]
    tags_df.loc[index_df['tag'] == 'training', 'tag'] = 0
    tags_df.loc[index_df['tag'] == 'validation', 'tag'] = 0
    tags_df.loc[index_df['tag'] == 'testing', 'tag'] = 1
    tags_df.columns = ['image', 'mask', 'dataset']
    tags_df.to_csv(tags_csv, index=False, encoding='utf-8')


def main(args):
    models = [n.strip() for n in args.models.split(',')]
    before_run(args.project_id, args.feature_csv, args.target_csv, args.tags_csv)
    learn.main(args.feature_csv, args.target_csv, args.tags_csv, models, args.output_dir, args.cv, args.auto_opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_csv', help='feature csv file', default='./example-debug/output/feature_selected.csv')
    parser.add_argument('--target_csv', help='target csv file', default='./example-debug/target.csv')
    parser.add_argument('--tags_csv', help='tags', default='./example-debug/tags.csv')
    parser.add_argument('--output_dir', help='output csv file', default='./example-debug/output')
    parser.add_argument('--models', help='models', default='knn, bayes, xgboost, deep, svm, logistic, decision_tree, random_forest')
    parser.add_argument('--cv', help='number of cross validation', type=int, default=5)
    parser.add_argument('--auto_opt', help="auto optimization", action='store_true', default=False)
    parser.add_argument('--project_id', help="learn task project id")
    args = parser.parse_args()
    main(args)
