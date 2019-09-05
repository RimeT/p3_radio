#!/usr/bin/env python2

import argparse
import pandas as pd
import json

from scholar.schema import schema
from scholar.schema import radiomics as projects
from scholar.radiomics.data import new_feature, reset_target_feature, load_feature_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import data to radiomics project')
    parser.add_argument('--project-id', type=str,
                        help='Identifier of a radiomics project')
    parser.add_argument('--data-path', type=str,
                        help='Path of JSON encoded csv data')
    parser.add_argument('--key-name', type=str,
                        help='Join by key name')
    args = parser.parse_args()

    project_id = args.project_id

    with open(args.data_path) as f:
        data = json.load(f)

    kn = args.key_name

    project, = projects[schema.projectId == project_id]
    features = project >> 'data'

    # XXX clear old custom data
    reset_target_feature(features)
    for feature in features:
        if feature['group'] == 'custom':
            feature.to_shadow()

    feature, = features[schema['name'] == kn]
    feature_df = load_feature_data(feature)
    custom_df = pd.DataFrame(data['data'], columns=[k['name'] for k in data['columns']])

    # force join with str
    feature_df[kn] = feature_df[kn].astype(str)
    custom_df[kn] = custom_df[kn].astype(str)

    # manual left join keeps row number of left table, no pd.merge here
    df = pd.DataFrame(columns=[kn] + [col for col in custom_df if col != kn])
    df[kn] = feature_df[kn]
    for i, row in custom_df.iterrows():
        for col in custom_df:
            if col == kn:
                continue
            df[col][feature_df[kn] == row[kn]] = row[col]

    new_features = []
    feature_names = [k['name'] for k in features.nodes()]
    for k in data['columns']:
        col = k['name']
        if col == kn:
            continue
        col_name = col
        for i in xrange(100):
            if col_name in feature_names:
                col_name = col + '_' + str(i + 1)
            else:
                break
        else:
            raise Exception("Column name '{}' conflict.".format(col))
        feature = new_feature(name=col_name,
                              group=k.get('group', 'custom'),
                              type=k.get('type', 'category'),
                              intent=k.get('intent', 'in'),
                              data=df[col])
        new_features.append(feature)

    for f in new_features:
        features << f

    df = df.replace({pd.np.nan: None})
