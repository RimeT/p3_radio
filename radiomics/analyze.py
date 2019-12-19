import argparse
import json
import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tools
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# 2D
# parser.add_argument('--feature_csv', help='feature csv file', default='./example-2D/output/filter/feature_selected.csv')
# parser.add_argument('--target_csv', help='target csv file', default='./example-2D/target.csv')
# parser.add_argument('--output_dir', help='output csv file', default='./example-2D/output/analysis')

# 3D
# parser.add_argument('--feature_csv', help='feature csv file', default='./example-3D/output/filter/feature_selected.csv')
# parser.add_argument('--target_csv', help='target csv file', default='./example-3D/label_N_4.csv')
# parser.add_argument('--output_dir', help='output csv file', default='./example-3D/output/analysis')

# fuwai
# parser.add_argument('--feature_csv', help='feature csv file', default='./example-fuwai/output/filter/feature_selected.csv')
# parser.add_argument('--target_csv', help='target csv file', default='./example-fuwai/target.csv')
# parser.add_argument('--output_dir', help='output csv file', default='./example-fuwai/output/analysis')


def main(df_path, target_path, output_path, k):
    feature_df = pd.read_csv(df_path)
    label_df = pd.read_csv(target_path)
    columns = feature_df.columns
    key_columns = [x for x in columns if x in tools.keywords]
    feature_columns = [x for x in columns if x not in tools.keywords]

    feature_df, label_df = tools.prepare_feature_n_label(feature_df, label_df)

    df_analysis = feature_df[feature_columns]
    df_analysis = tools.preprocessing(df_analysis)
    df_norm = tools.scale_on_feature(df_analysis)

    target = label_df[['label']]
    label = target.label
    class_nb = len(set(target.label.tolist()))

    le = LabelEncoder().fit(target)
    el = le.transform(target)

    # corr
    corr = df_analysis.corr()
    corr.to_csv(os.path.join(output_path, 'corr.csv'), encoding='utf-8')
    # fig, ax = plt.subplots(figsize=(8, 5))
    # fig, ax = plt.subplots()
    plt.figure(figsize=(8, 5))
    # draw corr

    import seaborn as sns
    sns.set()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap,
                # ax=ax,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                # xticklabels=range(1, 1+len(corr.columns)),
                # yticklabels=range(1, 1+len(corr.columns)),
                mask=mask)
    plt.savefig(os.path.join(output_path, "correlation.png"), bbox_inches='tight', dpi=600)
    plt.show()
    plt.clf()

    # PCA
    pca_feature = df_norm.copy()

    pca_components = min(pca_feature.shape[0], pca_feature.shape[1], 10)
    pca = PCA(n_components=pca_components)
    pca.fit(pca_feature)

    pc_names = ['pc' + str(p) for p in range(1, pca_components + 1)]
    variance = pca.explained_variance_ratio_
    variance_percent = np.round(variance, decimals=3) * 100

    # save pca
    pca_df = []
    for idx, v in enumerate(variance):
        pca_df += [{'pca': 'pc' + str(idx), 'percent': v}]
    pca_df = pd.DataFrame(pca_df)
    pca_df.to_csv(os.path.join(output_path, 'pca.csv'), index=False, encoding='utf-8')

    # raw data
    raw_pca = pd.DataFrame(np.transpose(pca.components_, (1, 0)),
                           columns=["pc_" + str(c) for c in range(len(pca.components_))])
    raw_pca.to_csv(os.path.join(output_path, "raw_analyze_pca.csv"), index=False, encoding='utf-8')

    plt.ylabel('% Variance')
    plt.xlabel('Principal Features')
    plt.title('PCA Analysis')
    plt.ylim(0.0, 100)
    plt.bar(np.arange(pca_components), variance_percent, align='center')
    plt.xticks(np.arange(pca_components), pc_names)
    plt.plot(np.cumsum(variance_percent))
    plt.savefig(os.path.join(output_path, "pca.png"), dpi=600)
    plt.show()

    print("PCA success")

    # STATISTICS
    stats = []
    null_columns = list(compress(feature_columns, list(df_analysis.isnull().any())))

    for c in feature_columns:
        v_avg = np.mean(df_analysis[c])
        v_std = np.std(df_analysis[c])
        v_max = np.max(df_analysis[c])
        v_min = np.min(df_analysis[c])
        stats += [{
            'name': c,
            'type': 'Real',
            'missing': 1 if c in null_columns else 0,
            'mean': v_avg,
            'std': v_std,
            'max': v_max,
            'min': v_min
        }]
    stats_df = pd.DataFrame(stats)[['name', 'type', 'missing', 'max', 'min', 'mean', 'std']]
    stats_df.to_csv(os.path.join(output_path, 'stat.csv'), index=False, encoding='utf-8')
    print("Statistics success")

    # Cluster
    kmeans = KMeans(n_clusters=k, random_state=42)
    values = df_norm.values
    kmeans.fit(values)
    cluster_labels = kmeans.labels_
    cluster_labels_str = ["cluster" + str(x + 1) for x in cluster_labels]

    # cluster results
    cluster_res_df = feature_df.copy()
    cluster_res_df['class'] = cluster_labels_str
    cluster_res_df = cluster_res_df[
        ['class', 'image', 'mask'] + [x for x in cluster_res_df.columns if x not in ['class', 'mask', 'image']]]
    cluster_res_df.to_csv(os.path.join(output_path, 'cluster_voi.csv'), index=False, encoding='utf-8')

    # cluster centroids
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = pd.DataFrame(cluster_centers, columns=feature_columns)
    cluster_centers['class'] = ["cluster" + str(x + 1) for x in cluster_centers.index.tolist()]
    cluster_centers = cluster_centers[['class'] + [x for x in cluster_centers.columns if x not in ['class']]]
    cluster_centers.to_csv(os.path.join(output_path, 'cluster_centroids.csv'), index=False, encoding='utf-8')
    cluster_centers_class = cluster_centers['class']
    cluster_aval_columns = cluster_centers.columns[1:]
    # print(cluster_aval_columns, type(cluster_aval_columns.tolist()))
    cluster_nparr = cluster_centers[cluster_aval_columns].values
    cluster_out_fig = os.path.join(output_path, 'cluster_curve.png')
    tools.curve_with_xlabels(cluster_nparr, cluster_aval_columns.tolist(), cluster_centers_class.tolist(),
                             save_path=cluster_out_fig)
    if os.path.isfile(cluster_out_fig):
        pass

    # cluster statistics
    cluster_count = cluster_res_df[['class']].groupby('class').size()
    cluster_stat_json = dict()

    distance_df = cluster_res_df.copy()
    distances = []
    for idx, r in distance_df.iterrows():
        centroid = cluster_centers.loc[cluster_centers['class'] == r['class']].iloc[0, 1:].values
        dis = distance.euclidean(df_norm.iloc[idx, :].values, centroid)
        distances += [dis]
    distance_df['distance'] = distances
    distance_avg = distance_df[['class', 'distance']].groupby('class')['distance'].mean()

    for idx, c in cluster_count.items():
        stat_dict = dict()
        stat_dict['number'] = c
        stat_dict['percent'] = float(c) / float(df_norm.shape[0])
        stat_dict['avg_distance'] = distance_avg[idx]
        cluster_stat_json[idx] = stat_dict

    with open(os.path.join(output_path, 'cluster_stat.json'), 'w') as fp:
        json.dump(cluster_stat_json, fp, indent=4, sort_keys=True)

    print("Clustering success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # debug
    parser.add_argument('--feature_csv', help='feature csv file', default='./example-debug/output/feature_selected.csv')
    parser.add_argument('--target_csv', help='target csv file', default='./example-debug/target.csv')
    parser.add_argument('--output_dir', help='output csv directory', default='./example-debug/output')
    parser.add_argument('--k', help='number of clusters', type=int, default=4)
    args = parser.parse_args()
    main(args.feature_csv, args.target_csv, args.output_dir, args.k)
