# -*- coding: UTF-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np
import os
import json
from itertools import cycle
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import argparse
import tools
from sklearn.utils import shuffle


def main(df_path, target_path, output_path, filters):
    feature_classes = ['glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm', 'shape', 'firstorder']
    
    df = pd.read_csv(df_path)
    label_df = pd.read_csv(target_path)
    
    df, label_df = tools.prepare_feature_n_label(df, label_df)
    
    columns = df.columns
    key_columns = [x for x in columns if x in tools.keywords]
    
    feature_columns = [x for x in columns if x not in tools.keywords]
    
    df_selected = df[feature_columns]
    df_selected = tools.preprocessing(df_selected)
    df_selected = tools.scale_on_min_max(df_selected, feature_range=(0, 1))
    
    extracted_classes = [x for x in columns if len([y for y in feature_classes if y in x[:len(y)]]) > 0]
    valid_feature_classes = set([n.split("_")[0] for n in extracted_classes])
    custom_classes = [x for x in columns if x not in extracted_classes and x not in tools.keywords]
    # all_classes = custom_classes + extracted_classes
    
    target = label_df[['label']]
    label = target.label
    class_nb = len(set(target.label.tolist()))
    
    le = LabelEncoder().fit(target)
    el = le.transform(target)
    
    print('Preprocess success')
    
    
    def fs_lasso(x, y):
    
        y = y * 1.0
        features = x.columns
    
        x, y = shuffle(x, y, random_state=55)
    
        skf = StratifiedKFold(n_splits=5, random_state=2).split(x, y)
    
        # cv - best alpha
        model = LassoCV(cv=skf, alphas=np.linspace(1e-5, 0.1, 1000)).fit(x, y)
        m_mses = model.mse_path_
        m_log_alphas = -np.log10(model.alphas_)
        best_alpha = -np.log10(model.alpha_)
        # plot
        plt.figure(2)
        plt.plot(m_log_alphas, m_mses, ':')
        plt.plot(m_log_alphas, m_mses.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
        plt.axvline(best_alpha, linestyle='--', color='k', label='alpha: CV estimate')
        plt.legend()
        plt.xlabel('-Log(alpha)')
        plt.ylabel('Mean square error')
        plt.title('Mean square error paths on each fold')
        plt.axis('tight')
        plt.savefig(os.path.join(output_path, "lasso_mse.png"), dpi=600)
        plt.show()
    
        # lasso path
        print("Computing regularization path using the lasso...")
        alphas_lasso, coefs_lasso, _ = lasso_path(x, y,
                                                  fit_intercept=False,
                                                  alphas=np.linspace(model.alpha_, 0.5, 100))
        # plot
        plt.figure(1)
        neg_log_alphas_lasso = -np.log10(alphas_lasso)
        for coef_l in coefs_lasso:
            plt.plot(neg_log_alphas_lasso, coef_l)
        plt.xlabel('-Log(alpha)')
        plt.ylabel('coefficients')
        plt.title('Lasso Paths')
        plt.axis('tight')
        plt.savefig(os.path.join(output_path, "lasso_path.png"), dpi=600)
        plt.show()
    
        mask_cv = model.coef_ != 0
        mask_cv = [i for i, m in enumerate(mask_cv) if m == True]
    
        # save alpha
        with open(os.path.join(output_path, 'lasso_alpha.json'), 'w') as fp:
            json.dump({"alpha": best_alpha}, fp, indent=4, sort_keys=True)
    
        # save mse path
        mses = []
        for c in range(m_mses.shape[1]):
            path_value = zip(list(m_log_alphas), list(m_mses[:, c]))
            path_value = list(path_value) if type(path_value) is not list else path_value
            mses += [{
                "name": "fold" + str(c),
                "path": path_value
            }]
        with open(os.path.join(output_path, 'lasso_mse.json'), 'w') as fp:
            json.dump(mses, fp, indent=4, sort_keys=True)
    
        # save lasso path
        lasso_paths = []
        for f in range(coefs_lasso.shape[0]):
            path_value = zip(list(neg_log_alphas_lasso), list(coefs_lasso[f, :]))
            path_value = list(path_value) if type(path_value) is not list else path_value
    
            if not list(set([abs(b) for b in list(coefs_lasso[f, :])])) == [0]:
                res = [{
                    "name": features[f],
                    "path": path_value
                }]
                lasso_paths += res
        with open(os.path.join(output_path, 'lasso_path.json'), 'w') as fp:
            json.dump(lasso_paths, fp, indent=4, sort_keys=True)
    
        # raw data
        chosen = [True if c in mask_cv else False for c in range(len(model.coef_))]
        raw_lasso = pd.DataFrame(zip(features, model.coef_, chosen), columns=["features", "coef", "chosen"])
        raw_lasso.to_csv(os.path.join(output_path, "raw_filter_lasso.csv"), index=False, encoding='utf-8')
    
        return x[features[mask_cv]]
    
    
    def fs_vt(data):
        try:
            selector = VarianceThreshold(0.02)
            selector.fit(data)
        except Exception:
            try:
                selector = VarianceThreshold(0.01)
                selector.fit(data)
            except:
                try:
                    selector = VarianceThreshold(0.005)
                    selector.fit(data)
                except:
                    selector = VarianceThreshold(0)
                    selector.fit(data)
        # raw data
        variances = selector.variances_
        chosen = [True if c in selector.get_support(indices=True) else False for c in range(len(data.columns))]
        raw_variance = pd.DataFrame(zip(data.columns, variances, chosen), columns=["features", "variance", "chosen"])
        raw_variance.to_csv(os.path.join(output_path, "raw_filter_variance.csv"), index=False, encoding='utf-8')
    
        return data[data.columns[selector.get_support(indices=True)]]
    
    
    def fs_kbest(x, y, k=300, score_method=f_classif):
    
        if k > x.shape[1]:
            k = x.shape[1]
    
        columns = x.columns
        selector = SelectKBest(score_method, k=k)
        selector.fit(x, y)
    
        # save p-values
        p_values = selector.pvalues_
        p_dict = dict(zip(columns, p_values))
        with open(os.path.join(output_path, 'kbest_pvalues.json'), 'w') as fp:
            json.dump(p_dict, fp, indent=4, sort_keys=True)
    
        # raw data
        chosen = [True if c in selector.get_support(indices=True) else False for c in range(len(x.columns))]
        raw_kbest = pd.DataFrame(zip(x, p_values, chosen), columns=["features", "p_values", "chosen"])
        raw_kbest.to_csv(os.path.join(output_path, "raw_filter_kbest.csv"), index=False, encoding='utf-8')
        cols = selector.get_support(indices=True)
        return x[x.columns[cols]], p_dict
    
    
    vt_columns = []
    kbest_columns = []
    lasso_columns = []
    
    for f in filters:
        if 'best' in f:
            df_selected, p_dict = fs_kbest(df_selected, el)
            kbest_columns = df_selected.columns
    
            print('kbest success, feature left: ', df_selected.shape[1])
            pass
        elif 'lasso' in f:
            df_selected = fs_lasso(df_selected, el)
            lasso_columns = df_selected.columns
            print('lasso success, feature left: ', df_selected.shape[1])
            pass
        else:
            df_selected = fs_vt(df_selected)
            vt_columns = df_selected.columns
            vt_res = []
            print('variance success, feature left: ', df_selected.shape[1])
    
            for f in valid_feature_classes:
                nb_before = len([x for x in columns if f in x])
                nb_after = len([x for x in vt_columns if f in x])
                vt_res += [{'name': f, 'before': nb_before, 'after': nb_after}]
            if len(custom_classes) > 0:
                vt_res += [{'name': 'custom',
                            'before': len([x for x in columns if x in custom_classes]),
                            'after': len([x for x in vt_columns if x in custom_classes])
                            }]
    
            with open(os.path.join(output_path, 'variance.json'), 'w') as fp:
                json.dump(vt_res, fp, indent=4, sort_keys=True)
            pass
    
    
    selected = []
    
    for f in columns:
        selected += [{
            'feature': f,
            'variance': int(f in vt_columns),
            'k-best': int(f in kbest_columns),
            'lasso': int(f in lasso_columns)}
        ]
    
    res_df = pd.DataFrame(selected)
    res_df.to_csv(os.path.join(output_path, 'selection_result.csv'), index=False, encoding='utf-8')
    
    for c in key_columns:
        df_selected[c] = df[[c]]
    df_selected = df[key_columns + [c for c in df_selected if c not in key_columns]]
    df_selected.to_csv(os.path.join(output_path, 'feature_selected.csv'), index=False, encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 2D
    parser.add_argument('--feature_csv', help='feature csv file', default='./example-2D/output/feature_ria.csv')
    parser.add_argument('--target_csv', help='target csv file', default='./example-2D/target.csv')
    parser.add_argument('--filters', help='filters', default='variance, k-best, lasso')
    parser.add_argument('--output_dir', help='output dir', default='./example-2D/output/filter')
    
    # 3D
    # parser.add_argument('--feature_csv', help='feature csv file', default='./example-3D/output/feature_py.csv')
    # parser.add_argument('--target_csv', help='target csv file', default='./example-3D/label_N_4.csv')
    # parser.add_argument('--filters', help='filters', default='variance, k-best, lasso')
    # parser.add_argument('--output_dir', help='output dir', default='./example-3D/output/filter')
    
    # fuwai
    # parser.add_argument('--feature_csv', help='feature csv file', default='./example-fuwai/output/feature_ria.csv')
    # parser.add_argument('--target_csv', help='target csv file', default='./example-fuwai/target.csv')
    # parser.add_argument('--filters', help='filters', default='variance, k-best, lasso')
    # parser.add_argument('--output_dir', help='output dir', default='./example-fuwai/output/filter')
    
    # debug
    # parser.add_argument('--feature_csv', help='feature csv file', default='./example-debug/output/feature_ria.csv')
    # parser.add_argument('--target_csv', help='target csv file', default='./example-debug/target.csv')
    # parser.add_argument('--filters', help='filters', default='variance, k-best, lasso')
    # parser.add_argument('--output_dir', help='output dir', default='./example-debug/output')
    
    args = parser.parse_args()
    
    filters = args.filters.split(',')
    main(args.feature_csv, args.target_csv, args.output_dir, filters)
    