import argparse
import extract
import pandas as pd
import json

from scholar.schema import schema
from scholar.schema import radiomics as projects
from scholar.radiomics.data import new_feature, load_data_index, save_data_flags
from scholar.utils.strings import glob_match


def before_run(project_id, row_ids, data_csv_path):
    project, = projects[schema.projectId == project_id]
    features = project >> 'data'
    index_df = load_data_index(features).set_index('id')
    index_df['flag'] = 0 if row_ids is not None else 1
    index_df.loc[row_ids if row_ids else [], 'flag'] = 1
    data_csv = index_df.loc[index_df['flag'] == 1][['image', 'mask']]
    data_csv.to_csv(data_csv_path, index=False, encoding='utf-8')


def after_run(project_id, output_csv, row_ids):
    data = pd.read_csv(output_csv, header=0, index_col=False)
    project, = projects[schema.projectId == project_id]
    features = project >> 'data'

    images_df = pd.read_csv(features // 'images.csv', names=['_id', '_image', '_mask', '_tag', '_flag'], header=None, index_col=False)

    # delete old radiomics features
    del features[lambda x: glob_match('radiomics/**', x['group'])]

    save_data_flags(features, row_ids)
    df = images_df.join(data.set_index('mask'), on='_mask')

    for col in df.columns:
        if col in ['image', 'mask'] or str(col).startswith('_'):
            continue
        features << new_feature(
            name=col,
            data=df[col],
            group='/'.join(['radiomics'] + col.split('_')[:2]),
            type='number',
            intent='in')


def main(args):
    row_ids = None if args.row_ids == 'None' else json.loads(args.row_ids)
    before_run(args.project_id, row_ids, args.data_csv)
    extract.main(args.data_csv, args.output, args.lib.lower(), args.cpus, args.img_reader)
    after_run(args.project_id, args.output, row_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', help='image and mask columns', default='./example-debug/data.csv')
    parser.add_argument('--output', help='feature output folder', default='./example-debug/output/feature_py.csv')
    parser.add_argument('--lib', help='RIA or Pyradiomics', default='py')
    parser.add_argument('--cpus', help='cpu cores', type=int, default=8)
    parser.add_argument('--img_reader', help='dicom, nii, nrrd', default="dicom")
    parser.add_argument('--project_id', help='project id')
    parser.add_argument('--row_ids', help='row ids')
    args = parser.parse_args()
    main(args)
