# -*- coding: UTF-8 -*-
from __future__ import print_function

import argparse
import glob
import json
import math
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

import tools


def _array_nan2zero(arr):
    for idx, item in enumerate(arr):
        if math.isnan(item) or math.isinf(item):
            arr[idx] = 0.
    return arr


def _get_classifier(m, model_params, tv_data, is_binary, random_state=42, class_nb=None, auto_opt=False):
    tv_feature, tv_label = tv_data
    class_name = m
    if 'svc' == m:
        if auto_opt:
            param = dict(kernel=('linear', 'rbf'),
                         C=np.logspace(-4, 4, 20),
                         gamma=(1, 2, 3, 'auto'),
                         decision_function_shape=('ovo', 'ovr'),
                         shrinking=(True, False)
                         )
            clf_probe = RandomizedSearchCV(svm.SVC(), param, n_iter=200, scoring='accuracy', verbose=1,
                                           cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6,
                                           random_state=42)
            clf_probe.fit(tv_feature, tv_label)
            return svm.SVC(probability=True, random_state=random_state, **clf_probe.best_params_), class_name
        else:
            return svm.SVC(probability=True, **model_params), class_name
    elif 'nusvc' == m:
        if auto_opt:
            param = dict(kernel=('linear', 'rbf', 'linear'),
                         gamma=(1, 2, 3, 'auto'),
                         decision_function_shape=('ovo', 'ovr'),
                         shrinking=(True, False),
                         )
            clf_probe = RandomizedSearchCV(svm.SVC(), param, n_iter=200, scoring='accuracy', verbose=1,
                                           cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6,
                                           random_state=42)
            clf_probe.fit(tv_feature, tv_label)
            return svm.NuSVC(probability=True, random_state=random_state, **clf_probe.best_params_), class_name
        else:
            return svm.NuSVC(probability=True, **model_params), class_name
            # return svm.SVC(kernel='linear', probability=True, random_state=random_state), class_name
    elif 'svm' in m:
        # class_name = "SVM"
        if auto_opt:
            param = dict(kernel=('linear', 'rbf'),
                         C=np.logspace(-4, 4, 20),
                         gamma=(1, 2, 3, 'auto'),
                         decision_function_shape=('ovo', 'ovr'),
                         shrinking=(True, False)
                         )
            clf_probe = RandomizedSearchCV(svm.SVC(), param, n_iter=200, scoring='accuracy', verbose=1,
                                           cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6,
                                           random_state=42)
            clf_probe.fit(tv_feature, tv_label)
            return svm.SVC(probability=True, random_state=random_state, **clf_probe.best_params_), class_name
        else:
            return svm.SVC(kernel='linear', probability=True, random_state=random_state), class_name
    elif 'bayes' in m:
        # class_name = "Naive Bayes"
        return GaussianNB(), class_name
    elif 'knn' in m:
        # class_name = "KNN"
        if auto_opt:
            param = dict(n_neighbors=tuple(range(class_nb, 4 * class_nb))[1::2],
                         weights=('uniform', 'distance'),
                         algorithm=('auto', 'ball_tree', 'kd_tree', 'brute'),
                         leaf_size=tuple(range(1, 3))
                         )
            clf_probe = GridSearchCV(KNeighborsClassifier(), param, scoring='accuracy', verbose=1,
                                     cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6)
            clf_probe.fit(tv_feature, tv_label)
            return KNeighborsClassifier(**clf_probe.best_params_), class_name
        else:
            return KNeighborsClassifier(n_neighbors=class_nb * 3), class_name
    elif 'logistic' in m:
        # class_name = "Logistic Regression"
        if auto_opt:
            param = [{'penalty': ['l1', 'l2'],
                      'C': np.logspace(-4, 4, 20),
                      'solver': ['liblinear'],
                      'multi_class': ['ovr']},
                     {'penalty': ['l2'],
                      'C': np.logspace(-4, 4, 20),
                      'solver': ['lbfgs'],
                      'multi_class': ['ovr', 'multinomial']}]

            clf_probe = GridSearchCV(LogisticRegression(tol=1e-6, max_iter=1000), param, scoring='accuracy',
                                     verbose=1,
                                     cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6)
            clf_probe.fit(tv_feature, tv_label)
            return LogisticRegression(tol=1e-6, max_iter=1000, random_state=random_state,
                                      **clf_probe.best_params_), class_name
        else:
            return LogisticRegression(tol=1e-6, max_iter=1000, random_state=random_state,
                                      solver='liblinear'), class_name
    elif 'decision' in m:
        # class_name = "Decision Tree"
        if auto_opt:
            param = dict(max_features=['auto', 'sqrt'],
                         max_depth=[int(x) for x in np.linspace(20, 200, 10)] + [None],
                         min_samples_split=[2, 5, 10],
                         min_samples_leaf=[1, 5, 10, 20, 50, 100]
                         )
            clf_probe = RandomizedSearchCV(DecisionTreeClassifier(), param, n_iter=200, scoring='accuracy',
                                           verbose=2,
                                           cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6)
            clf_probe.fit(tv_feature, tv_label)
            return DecisionTreeClassifier(random_state=random_state, **clf_probe.best_params_), class_name
        else:
            return DecisionTreeClassifier(max_depth=200, min_samples_leaf=15, random_state=random_state), class_name
    elif 'random' in m:
        # class_name = "Random Forest"
        if auto_opt:
            param = dict(n_estimators=[int(x) for x in np.linspace(200, 1000, 10)],
                         max_features=['auto', 'sqrt'],
                         max_depth=[int(x) for x in np.linspace(10, 110, 11)] + [None],
                         min_samples_split=[2, 5, 10],
                         min_samples_leaf=[10, 20, 50, 100],
                         bootstrap=[True, False]
                         )
            clf_probe = RandomizedSearchCV(RandomForestClassifier(), param, n_iter=50, scoring='accuracy',
                                           verbose=2,
                                           cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6)
            clf_probe.fit(tv_feature, tv_label)
            return RandomForestClassifier(random_state=random_state, **clf_probe.best_params_), class_name
        else:
            return RandomForestClassifier(n_estimators=100, max_depth=100, random_state=random_state), class_name
    elif 'xgboost' in m:
        # class_name = "Gradient Boost Tree"
        if is_binary:
            return xgb.XGBClassifier(objective='binary:logistic', n_estimators=200, max_depth=10,
                                     random_state=random_state), class_name
        else:
            return xgb.XGBClassifier(objective='multi:softprob', n_estimators=200, max_depth=200,
                                     random_state=random_state, num_class=class_nb), class_name
    elif 'deep' in m:
        # class_name = "Deep Learning"
        if auto_opt:
            param = dict(alpha=10.0 ** -np.arange(1, 10),
                         activation=['tanh', 'relu'],
                         solver=['adam'],
                         learning_rate=['adaptive']
                         )
            clf_probe = GridSearchCV(MLPClassifier(max_iter=5000, tol=1e-5), param, scoring='accuracy', verbose=1,
                                     cv=StratifiedKFold(n_splits=3, random_state=42), n_jobs=6)
            clf_probe.fit(tv_feature, tv_label)
            return MLPClassifier(random_state=random_state, max_iter=5000, tol=1e-5,
                                 **clf_probe.best_params_), class_name
        else:
            return MLPClassifier(random_state=random_state, max_iter=5000, tol=1e-5,
                                 learning_rate='adaptive'), class_name
    return None, None


def test_analyze(clf, out_dir, df, df_learn, test_feature, encoded_test_label, label_encoder, is_binary,
                 origin_classes):
    predicts, proba, acc, class_report, auc_report, fpr, tpr, roc, samples = get_predict_report(clf,
                                                                                                test_feature,
                                                                                                encoded_test_label,
                                                                                                label_encoder,
                                                                                                is_binary,
                                                                                                origin_classes,
                                                                                                sess="test")

    _get_roc(fpr, tpr, out_dir)
    tools.save_json({'acc': acc}, os.path.join(out_dir, 'acc.json'))

    samples["mask"] = df[df_learn['dataset'] == 1]["mask"].tolist()
    samples["image"] = df[df_learn['dataset'] == 1]["image"].tolist()
    samples = samples[["image", "mask"] + [x for x in samples.columns if x not in ["image", "mask"]]]
    class_report.to_csv(os.path.join(out_dir, 'report.csv'), index=False, encoding='utf-8')
    samples.to_csv(os.path.join(out_dir, 'samples_result.csv'), index=False, encoding='utf-8')
    tools.save_json(roc, os.path.join(out_dir, 'roc.json'))


def _get_fold_folder(clf_name, fold, dataset=None, temp_dir=None):
    if dataset:
        return os.path.join(temp_dir, clf_name, 'fold_' + str(fold), dataset)
    else:
        return os.path.join(temp_dir, clf_name, 'fold_' + str(fold))


def _get_cv_folder(clf_name, dataset=None, temp_dir=None):
    if dataset:
        return os.path.join(temp_dir, clf_name, 'cv', dataset)
    else:
        return os.path.join(temp_dir, clf_name, 'cv')


def _get_roc(fpr, tpr, p):
    for origin_label in fpr:
        tools.roc_for_class([fpr[origin_label]],
                            [tpr[origin_label]],
                            class_name=origin_label,
                            save_path=os.path.join(p, 'roc_class_' + str(origin_label) + '.png'))


def get_predict_report(clf, x, y, label_encoder, is_binary, origin_classes, sess="train"):
    predicts = clf.predict(x)
    proba = clf.predict_proba(x)
    acc = clf.score(x, y)

    # class report
    if is_binary:
        report = classification_report(y, predicts,
                                       labels=label_encoder.transform(origin_classes),
                                       target_names=origin_classes,
                                       output_dict=True)
    else:
        report = classification_report(np.argmax(y, axis=1),
                                       np.argmax(predicts, axis=1),
                                       labels=np.argmax(label_encoder.transform(origin_classes), axis=1),
                                       target_names=origin_classes,
                                       output_dict=True)
    report = pd.DataFrame.from_dict(report, orient='index')
    report['class'] = report.index
    report.drop('micro avg', inplace=True)
    report.drop('macro avg', inplace=True)

    # roc and auc for each class
    fpr, tpr, thresh, roc_auc, roc = dict(), dict(), dict(), dict(), dict()
    auc_report = []

    used_origin_idex = [0, 1] if is_binary else [i for i in sorted(list(set(list(np.argmax(y, axis=1)))))]
    for i in used_origin_idex:
        origin_label = origin_classes[i]
        if is_binary:
            class_specificity = report.loc[[l for l in origin_classes if l != origin_label][0]]['recall']
            fpr[origin_label], tpr[origin_label], thresh[origin_label] = roc_curve(y, proba[:, i], pos_label=i)
        else:
            class_specificity = 0.
            fpr[origin_label], tpr[origin_label], thresh[origin_label] = roc_curve(y[:, i], proba[:, i])

        thresh[origin_label][0] = 1.
        auc_score = auc(fpr[origin_label], tpr[origin_label])
        roc[origin_label] = dict()
        roc[origin_label]['fpr'] = _array_nan2zero(fpr[origin_label].tolist())
        roc[origin_label]['tpr'] = _array_nan2zero(tpr[origin_label].tolist())
        roc[origin_label]['thresh'] = list(map(np.float64, list(thresh[origin_label])))
        roc[origin_label]['auc'] = np.nan_to_num(auc_score)

        auc_report += [{
            'class': origin_label,
            'AUC': auc_score,
            'Sensitivity': report.loc[origin_label]['recall'],
            'Specificity': class_specificity
        }]

    auc_report = pd.DataFrame(auc_report)
    auc_report = auc_report[['class'] + [c for c in auc_report.columns if c != 'class']]

    # samples
    indexs = np.arange(len(y))
    samples = pd.DataFrame(indexs, columns=['orders'])
    samples['predict'] = label_encoder.inverse_transform(predicts)
    samples['label'] = label_encoder.inverse_transform(y)
    samples['p_predicted'] = [max(p) for p in proba]
    for p in range(len(origin_classes)):
        samples.insert(loc=len(samples.columns),
                       column='p_' + str(origin_classes[p]),
                       value=proba[:, p])
    samples['correct'] = samples.apply(lambda row: str(row['predict']) == str(row['label']), axis=1)
    samples = samples[['label', 'predict', 'p_predicted'] +
                      ['p_' + str(origin_classes[p]) for p in range(len(origin_classes))] + ['correct']]
    return predicts, proba, acc, report, auc_report, fpr, tpr, roc, samples


def tv_analyze(clf, clf_name, x, y, fold, data_index, dataset, temp_dir, label_encoder, is_binary, origin_classes):
    fold_path = _get_fold_folder(clf_name, fold, dataset, temp_dir=temp_dir)
    tools.makedir_ignore(fold_path)
    predicts, proba, acc, class_report, auc_report, fpr, tpr, roc, samples = get_predict_report(clf, x, y,
                                                                                                label_encoder,
                                                                                                is_binary,
                                                                                                origin_classes)

    _get_roc(fpr, tpr, fold_path)

    tools.save_json({'acc': acc}, os.path.join(fold_path, 'acc.json'))
    class_report.to_csv(os.path.join(fold_path, 'class_report.csv'), index=False, encoding='utf-8')
    auc_report.to_csv(os.path.join(fold_path, 'auc_report.csv'), index=False, encoding='utf-8')
    tools.save_json(roc, os.path.join(fold_path, 'roc.json'))
    # samples.insert(loc=len(patients.columns), column='fold', value=[fold for _ in indexs])
    return fpr, tpr


def classification_train(clf, clf_name, summary_results, tv_feature, cv, tv_label, encoded_tv_label, output_path,
                         temp_dir, label_encoder, is_binary, origin_classes):
    cv_res = dict()

    sf = StratifiedKFold(n_splits=cv, random_state=88)

    for fold, c in enumerate(sf.split(tv_feature, tv_label)):

        t_index, v_index = c[0], c[1]
        tx, ty, vx, vy = tv_feature.iloc[t_index], encoded_tv_label[t_index], \
                         tv_feature.iloc[v_index], encoded_tv_label[v_index]

        fold_dataset_path = _get_fold_folder(clf_name, fold, temp_dir=temp_dir)
        tools.makedir_ignore(fold_dataset_path)

        # fit model
        clf.fit(tx, ty)

        # save model
        joblib.dump(clf, os.path.join(fold_dataset_path, 'model.joblib'))

        cv_res[fold] = {'train': {'x': tx, 'y': ty, 'idx': t_index},
                        'valid': {'x': vx, 'y': vy, 'idx': v_index}}

        for d in cv_res[fold]:
            fpr, tpr = tv_analyze(clf, clf_name, cv_res[fold][d]['x'], cv_res[fold][d]['y'], fold=fold,
                                  data_index=cv_res[fold][d]['idx'], dataset=d, temp_dir=temp_dir,
                                  label_encoder=label_encoder, is_binary=is_binary, origin_classes=origin_classes)
            cv_res[fold][d]['roc'] = dict()
            for l in origin_classes:
                cv_res[fold][d]['roc'][l] = dict()
                cv_res[fold][d]['roc'][l]['fpr'] = fpr[l]
                cv_res[fold][d]['roc'][l]['tpr'] = tpr[l]

    # mean cv report
    acc_cv = {'acc': {'train': 1., 'valid': 1.}}
    for d in ['train', 'valid']:
        acc = []
        class_report_cv = []
        for f in range(cv):
            fold_dataset_path = _get_fold_folder(clf_name, f, d, temp_dir=temp_dir)
            auc_report = pd.read_csv(os.path.join(fold_dataset_path, 'auc_report.csv'))
            class_report = pd.read_csv(os.path.join(fold_dataset_path, 'class_report.csv'))
            auc_report['class'] = auc_report['class'].astype(str)
            class_report['class'] = class_report['class'].astype(str)
            report_df = auc_report.merge(class_report, on='class')
            report_df = report_df[['class', 'AUC', 'recall', 'f1-score', 'precision']]
            report_df.set_index('class', inplace=True)
            class_report_cv += [report_df]
            # acc
            with open(os.path.join(fold_dataset_path, 'acc.json')) as jf:
                acc += [json.load(jf)['acc']]
        acc_cv['acc'][d] = np.mean(np.array(acc))
        cv_dataset_path = _get_cv_folder(clf_name, d, temp_dir=temp_dir)
        tools.makedir_ignore(cv_dataset_path)
        report_mean = pd.concat(class_report_cv).groupby(level=0).mean()
        report_mean.to_csv(os.path.join(cv_dataset_path, 'report_mean.csv'), encoding='utf-8')
    cv_path = _get_cv_folder(clf_name, temp_dir=temp_dir)
    pd.DataFrame.from_dict(acc_cv, orient='index').to_csv(os.path.join(cv_path, 'acc_mean.csv'), encoding='utf-8')

    # roc - cv
    for l in origin_classes:
        for d in ['train', 'valid']:
            fpr = [cv_res[k][d]['roc'][l]['fpr'] for k in cv_res.keys()]
            tpr = [cv_res[k][d]['roc'][l]['tpr'] for k in cv_res.keys()]
            tools.roc_for_cv(fpr, tpr, l,
                             os.path.join(_get_cv_folder(clf_name, d, temp_dir=temp_dir), str(l) + "_.png"))

    # update summary results
    acc_path = os.path.join(temp_dir, clf_name, 'cv', 'acc_mean.csv')
    acc_df = pd.read_csv(acc_path)

    summary_results += [{
        'model': clf_name,
        'train': acc_df['train'][0],
        'valid': acc_df['valid'][0]
    }]

    model_compare_df = pd.DataFrame(summary_results)
    model_compare_df.to_csv(os.path.join(output_path, 'model_compare.csv'), index=False, encoding='utf-8')
    model_compare_df.to_csv(os.path.join(temp_dir, 'model_compare.csv'), index=False, encoding='utf-8')

    print(clf_name + " success")


def learn(clf_names, models, models_params, summary_results, tv_feature, tv_label, is_binary, random_state, class_nb,
          auto_opt,
          encoded_tv_label, output_path, temp_dir, label_encoder, origin_classes, cv):
    for m in models:
        clf, clf_name = _get_classifier(m, models_params[m], (tv_feature, tv_label), is_binary, random_state,
                                        class_nb=class_nb, auto_opt=auto_opt)
        clf_names += [clf_name]
        if not is_binary:
            clf = OneVsRestClassifier(clf)
        classification_train(clf, clf_name, summary_results, tv_feature, cv=cv, tv_label=tv_label,
                             encoded_tv_label=encoded_tv_label, output_path=output_path, temp_dir=temp_dir,
                             label_encoder=label_encoder, is_binary=is_binary, origin_classes=origin_classes)


def testing(clf_names, output_path, temp_dir, label_encoder, df, df_learn, test_feature, encoded_test_label, is_binary,
            origin_classes):
    for clf_name in clf_names:
        res_path = os.path.join(output_path, "models", clf_name)
        tools.makedir_delete(res_path)
        best_fold = None
        best_indicator = 0
        for cr in glob.glob(os.path.join(temp_dir, clf_name) + '/fold_*/valid/acc.json'):
            acc_score = tools.load_json(cr)["acc"]
            if acc_score > best_indicator:
                best_indicator = acc_score
                best_fold = int(cr.split('fold_')[1].split('/')[0])

        model_path = _get_fold_folder(clf_name, best_fold, 'model.joblib', temp_dir=temp_dir)
        shutil.copy(model_path, res_path + '/model.joblib')
        np.save(res_path + '/encoder.npy', label_encoder.classes_)
        model = joblib.load(model_path)
        test_analyze(model, res_path, df, df_learn, test_feature, encoded_test_label, label_encoder, is_binary,
                     origin_classes)


def main(feature_path, target_path, tags_path, models, output_path, cv, auto_opt, cls_params):
    clf_names = []  # formal name
    warnings.filterwarnings('ignore')

    # TEMP_DIR = tempfile.mkdtemp()
    TEMP_DIR = os.path.join(output_path, "temp")
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    else:
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)

    df = pd.read_csv(feature_path)
    df = shuffle(df, random_state=88)
    target = pd.read_csv(target_path)
    tags = pd.read_csv(tags_path)

    df, target, tags = tools.prepare_feature_n_label(df, target, tags)

    # 构造数据集
    feature_columns = [x for x in df.columns if x not in tools.keywords]
    df_learn = df[feature_columns]
    df_learn = tools.preprocessing(df_learn)

    # 归一化
    std_scaler = StandardScaler()
    df_learn[df_learn.columns] = std_scaler.fit_transform(df_learn)
    joblib.dump(std_scaler, os.path.join(output_path, "scalar.joblib"))

    df_learn['label'] = target['label'].tolist()
    df_learn['dataset'] = tags['dataset'].tolist()
    df_learn = df_learn[['label', 'dataset'] + feature_columns]

    # 训练集
    tv_df = df_learn[df_learn['dataset'] == 0]
    tv_label = [str(l) for l in tv_df.label.to_list()]
    tv_feature = tv_df[feature_columns]

    # 测试集
    test_df = df_learn[df_learn['dataset'] == 1]
    test_label = [str(l) for l in test_df.label.to_list()]
    test_feature = test_df[feature_columns]

    class_nb = len(set(df_learn.label.tolist()))
    is_binary = class_nb == 2

    # Iris 数据集，用于baseline测试
    enable_test = False
    if enable_test:
        iris = datasets.load_iris()
        iris_X, iris_y = iris.data, iris.target
        columns = [str(s) for s in range(iris_X.shape[1])]
        skf = StratifiedKFold(n_splits=5).split(iris_X, iris_y)
        iris_tv, iris_test = skf.next()
        tv_df = pd.DataFrame(iris_X[iris_tv], columns=columns)
        test_df = pd.DataFrame(iris_X[iris_test], columns=columns)
        class_nb = len(set(iris_y))
        tv_label = iris_y[iris_tv]
        test_label = iris_y[iris_test]
        tv_feature = tv_df[[x for x in columns if x not in tools.keywords]]
        test_feature = test_df[[x for x in columns if x not in tools.keywords]]

    # label encoding
    label_encoder, encoded_label, _ = tools.encode_b(tv_label + test_label) if not is_binary else tools.encode_l(
        tv_label + test_label)
    encoded_tv_label = encoded_label[0: len(tv_label)]
    encoded_test_label = encoded_label[len(tv_label):]

    origin_classes = list(label_encoder.classes_)
    origin_classes_tv = sorted(list(set(tv_label)))
    origin_classes_test = sorted(list(set(test_label)))

    # 结果总表
    summary_results = []
    random_state = 42

    learn(clf_names, models, cls_params, summary_results, tv_feature, tv_label, is_binary=is_binary,
          random_state=random_state,
          class_nb=class_nb, auto_opt=auto_opt, encoded_tv_label=encoded_tv_label, output_path=output_path,
          temp_dir=TEMP_DIR, label_encoder=label_encoder, origin_classes=origin_classes, cv=cv)
    testing(clf_names, output_path=output_path, temp_dir=TEMP_DIR, label_encoder=label_encoder, df=df,
            df_learn=df_learn, test_feature=test_feature, encoded_test_label=encoded_test_label, is_binary=is_binary,
            origin_classes=origin_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # fuwai
    # parser.add_argument('--feature_csv', help='feature csv file', default='./example-fuwai/output/filter/feature_selected.csv')
    # parser.add_argument('--target_csv', help='target csv file', default='./example-fuwai/target.csv')
    # parser.add_argument('--tags_csv', help='tags', default='./example-fuwai/tags.csv')
    # parser.add_argument('--output_dir', help='output csv file', default='./example-fuwai/output/')

    # debug
    parser.add_argument('--feature_csv', help='feature csv file', default='./example-debug/output/feature_selected.csv')
    parser.add_argument('--target_csv', help='target csv file', default='./example-debug/target.csv')
    parser.add_argument('--tags_csv', help='tags', default='./example-debug/tags.csv')
    parser.add_argument('--output_dir', help='output csv file', default='./example-debug/output')

    # 3D 模型测试
    # parser.add_argument('--feature_csv', help='feature csv file', default='./example-3D/output/filter/feature_selected.csv')
    # parser.add_argument('--target_csv', help='target csv file', default='./example-3D/label_N_4.csv')
    # parser.add_argument('--tags_csv', help='tags', default='./example-3D/tags.csv')
    # parser.add_argument('--output_dir', help='output csv file', default='./example-3D/output/')

    # 2D 模型测试
    # parser.add_argument('--feature_csv', help='feature csv file', default='./example-2D/output/filter/feature_selected.csv')
    # parser.add_argument('--target_csv', help='target csv file', default='./example-2D/target2.csv')
    # parser.add_argument('--tags_csv', help='tags', default='./example-2D/tags2.csv')
    # parser.add_argument('--output_dir', help='output csv file', default='./example-2D/output/learn')

    # parser.add_argument('--models', help='models', default='knn')
    parser.add_argument('--models', help='models',
                        default='knn, bayes, xgboost, deep, svm, logistic, decision_tree, random_forest')
    parser.add_argument('--cv', help='number of cross validation', type=int, default=5)
    parser.add_argument('--auto_opt', help="auto optimization", action='store_true', default=False)
    parser.add_argument('--cls_params', help="classifier parameters in json format", default=None, type=json.loads)

    args = parser.parse_args()

    # debug
    cls_params = args.cls_params
    if cls_params is None:
        raise ValueError("cls_params is None")

    # models = [n.strip() for n in args.models.split(',')]
    models = list(cls_params.keys())
    main(args.feature_csv, args.target_csv, args.tags_csv, models, args.output_dir, args.cv, args.auto_opt, cls_params)
