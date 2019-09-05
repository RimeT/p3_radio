#!/usr/bin/env python2

import os
import argparse
import pandas as pd
import cv2
import json
import numpy as np
import SimpleITK as sitk
import pydicom as dicom
import signal
import multiprocessing

from scholar.schema import schema
from scholar.schema import radiomics as projects
from scholar.radiomics.data import new_feature

DICOM_ATTRIBUTES = ['age', 'modality', 'patientId', 'seriesId', 'sex', 'studyId']


def main(base_paths, output_dir, project_id, label_defs, labelset_ids, pool):

    project, = projects[schema.projectId == project_id]
    features = project >> 'data'

    print 'processing {} units.'.format(len(base_paths))

    indices = []
    attributes = []
    annos = []
    reader = sitk.ImageSeriesReader()

    for n, base_path in enumerate(base_paths, 1):

        if not all(os.path.exists(os.path.join(base_path, chk)) for chk in ['regionfile', 'label.json']):
            # skip item without regionfile and interests
            continue

        output_path = os.path.join(output_dir, os.path.basename(base_path))
        print "generating '{}'...".format(output_path)
        try:
            os.makedirs(output_path)
        except OSError:
            pass

        # read DICOM series
        with open(os.path.join(base_path, 'images.lst')) as lst:
            img_filenames = [os.path.join(base_path, p.strip()) for p in lst if p.strip()]
            itk_filenames = reader.GetGDCMSeriesFileNames(os.path.dirname(img_filenames[0]))

        w, h = dicom.dcmread(itk_filenames[0]).pixel_array.shape

        # convert slices to nifti volume
        # v_data = np.stack([im.pixel_array for im in images])
        # img = sitk.GetImageFromArray(v_data)
        # img_path = os.path.join(output_path, 'image.nii')
        img_path = os.path.dirname(itk_filenames[0])
        # sitk.WriteImage(img, img_path, False)

        def find_itk_index(img_filename):
            try:
                return itk_filenames.index(img_filename)
            except:
                return -1

        slice_map = {i: find_itk_index(img_filename) for i, img_filename in enumerate(img_filenames)}

        print "convert complete."

        # read regionfile
        with open(os.path.join(base_path, 'regionfile')) as rgn_f:
            regionfile = json.load(rgn_f)

        # convert regionfile to lesion-based masks
        mask_path = os.path.join(output_path, 'mask')
        try:
            os.makedirs(mask_path)
        except OSError:
            pass

        print 'generating {} masks in regionfile...'.format(len(regionfile))

        res = pool.map_async(generate_mask, (((len(itk_filenames), w, h), slice_map, region, mask_path) for region in regionfile))
        mask_paths = res.get()

        print '{}/{} masks generated.'.format(len([p for p in mask_paths if p]), len(regionfile))

        # read metadata
        with open(os.path.join(base_path, '_data.json')) as meta_f:
            metadata = json.load(meta_f)
            dicom_metadata = {(k[0].upper() + k[1:]): v for k, v in metadata.iteritems() if k in DICOM_ATTRIBUTES}

        # read label data
        with open(os.path.join(base_path, 'label.json')) as lbl_f:
            # get interests filtered by specified labelset_ids
            interests = [desc
                         for desc in json.load(lbl_f)
                         if len(labelset_ids - set([str(label['labelSetId']) for label in desc['labels']])) == 0]
        for desc in interests:
            if os.getenv('DMS_TEST_MODE'):
                print 'anno: ' + json.dumps(desc)
            region_mask = os.path.join(mask_path, '{}.nii.gz'.format(desc['regionId']))
            if region_mask not in mask_paths:
                # failed to generate this region mask
                continue
            indices.append([len(indices), img_path, region_mask, metadata['tag'], 1])
            attributes.append(dicom_metadata)
            annos.append({
                label_defs[anno['labelSetId']]['name']: label_defs[anno['labelSetId']]['labels'][anno['labelId']]['name']
                for anno in desc['labels'] if str(anno['labelSetId']) in labelset_ids})

        print 'progress: {:.2f}%'.format(n * 100. / len(base_paths))

    # re-create feature folder
    features.to_shadow()
    features = project >> 'data'

    # save images.csv for indexing
    pd.DataFrame(indices).to_csv(features // 'images.csv', header=False, index=False, encoding='utf-8')

    df = pd.DataFrame(attributes)
    for col in df.columns:
        features << new_feature(name=col,
                                data=df[col],
                                group='general/image',
                                type='object',
                                intent='in')

    if annos:
        df = pd.DataFrame(annos)
        for col in df.columns:
            features << new_feature(name=col,
                                    data=df[col],
                                    group='general/label',
                                    type='category',
                                    intent='in')

    print 'completed'


def generate_mask((shape, slice_map, region, output_path)):
    mask = np.zeros(shape, dtype=np.int16)
    for contour in region['data']:
        try:
            itk_index = slice_map[contour['index']]
            if itk_index == -1:
                continue
            cv2.drawContours(mask[itk_index],
                             [np.array(contour['path']).astype(int)], 0,
                             1,
                             thickness=-1)
        except Exception:
            import traceback
            traceback.print_exc()
            return
    mask_img = sitk.GetImageFromArray(mask)
    mask_path = os.path.join(output_path, '{}.nii.gz'.format(region['regionId']))
    sitk.WriteImage(mask_img, mask_path, True)
    return mask_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import data to radiomics project')
    parser.add_argument('--project-id', type=str,
                        help='Identifier of a radiomics project')
    parser.add_argument('--output-dir', type=str,
                        help='Path of the directory to extract data to')
    parser.add_argument('--data-dir', type=str,
                        help='Path to the directory containing dataset download data')
    parser.add_argument('--labelset-ids', type=str,
                        help='List of label sets to import')
    args = parser.parse_args()

    base_paths = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)]
    base_paths = [d for d in base_paths if os.path.isdir(d)]
    output_dir = args.output_dir

    with open(os.path.join(args.data_dir, 'labels.json')) as f:
        label_defs = {
            label_set['labelSetId']: {
                'name': label_set['name'],
                'labels': {
                    label['labelId']: label for label in label_set['labels']
                }
            } for label_set in json.load(f)
        }
    labelset_ids = set(args.labelset_ids.split(','))

    n_threads = multiprocessing.cpu_count()

    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(n_threads)
    signal.signal(signal.SIGINT, original_sigint_handler)
    print 'initialized {} threads.'.format(n_threads)

    try:
        main(base_paths, output_dir, args.project_id, label_defs, labelset_ids, pool)
    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
    pool.join()
