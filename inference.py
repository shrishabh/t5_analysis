import pandas as pd
import json
from tqdm import tqdm
import sys
import argparse
from T5Wrapper import *
from sklearn import metrics

# from T5Wrapper import T5Wrapper


"""
Script to perform inference with T5 classifiers on Policy classification tasks. 
"""

def load_model(path):
    model = T5Wrapper()
    model.load_model("t5", path, use_gpu=True)
    return model

def get_predicted_df(test_set, model):
    #test_set = pd.read_json(test_path, orient='columns')
    predicted = []
    predicted_score = []
    with tqdm(total=test_set.shape[0]) as pbar:
        for idx, row in test_set.iterrows():
            pbar.update(1)
            text = row['source_text']
            #score = model.predict_batch(text)
            score = model.predict(text, labels=['Yes', 'No'])
            # print(score)
            predicted.append(score[0])
            predicted_score.append(score[1])
    test_set['predicted'] = predicted
    # print(predicted_score)
    # print(predicted)
    test_set['predicted_score'] = predicted_score
    #print(test_set)
    return test_set


def get_accuracies(test_set):
    pred_df = pd.DataFrame()
    pred_df['text'] = test_set.source_text
    pred_df['prediction'] = test_set.predicted.apply(lambda x: x.split(', '))
    pred_df = pred_df.explode('prediction', ignore_index=True)

    pred_df_grp = pred_df.groupby('prediction')

    pred_dic = {k: g["text"].tolist() for k, g in pred_df_grp}

    exploded_test = pd.DataFrame()

    exploded_test['text'] = test_set.source_text
    exploded_test['target'] = test_set.target_text

    exploded_test['target'] = exploded_test.target.apply(lambda x: x.split(', '))

    exploded_test = exploded_test.explode('target', ignore_index=True)

    grps = exploded_test.groupby('target')

    print("The support for each class in the set:")
    for name, df in grps:
        print(name, len(df))

    print("*" * 50)
    print()

    true_dic = {k: g["text"].tolist() for k, g in grps}
    flag = False
    for i in pred_dic:
        if i not in true_dic:
            print("{} not in True dic".format(i))
            flag = True
    if flag:
        print("Check the error")
        return None, None, None
    instances = {}

    for key in pred_dic:
        true = set(true_dic[key])
        pred = set(pred_dic[key])

        tp = true.intersection(pred)
        fn = true.difference(pred)
        fp = pred.difference(true)
        instances[key] = {
            'true_positive': tp,
            'false_negative': fn,
            'false_positive': fp
        }
        ln = len(true_dic[key])
        print("Category: {}, Total instances: {}".format(key, len(true_dic[key])))
        print("""True Positive: {} \nFalse Negative: {} \nFalse Positive: {}""".format(len(tp), len(fn), len(fp)))
        print("--" * 30)
        print("Precision: {} \t Recall: {}".format(len(tp) / (len(tp) + len(fp)), len(tp) / (len(tp) + len(fn))))
        print("===" * 30)
        print()
    return true_dic, pred_dic, instances


def get_results(test_bin):
    test_bin['predicted'] = test_bin['predicted'].apply(lambda x: x[0])
    #print(test_bin['predicted'])
    true_df = test_bin[test_bin['target_text'] == test_bin['predicted']]
    false_df = test_bin[test_bin['target_text'] != test_bin['predicted']]

    tp_df = true_df[true_df['target_text'] == 'Yes']
    tn_df = true_df[true_df['target_text'] == 'No']
    fp_df = false_df[false_df['target_text'] == 'No']
    fn_df = false_df[false_df['target_text'] == 'Yes']
    all_true = test_bin[test_bin['target_text'] == "Yes"]
    true_cats = {k: g["source_text"].tolist() for k, g in all_true.groupby('category')}
    true_cat_count = {k: len(v) for k, v in true_cats.items()}
    fn_dic = {k: g["source_text"].tolist() for k, g in fn_df.groupby('category')}
    fp_dic = {k: g["source_text"].tolist() for k, g in fp_df.groupby('category')}

    tp_grps = tp_df.groupby('category')

    results_dic = {}
    for name, df in tp_grps:
        tp = len(df)
        fp = len(fp_dic.get(name, []))
        fn = len(fn_dic.get(name, []))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        results_dic[name] = {
            'precision': precision,
            'recall': recall,
            'f1-score': get_f1(precision, recall),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'total_instances': true_cat_count[name]
        }

    ## get auc
    # all_grps = test_bin.groupby('category')

    # for name, df in all_grps:
    #     target_label = df['target_text'].tolist()
    #     predicted_score_ori = df['predicted_score'].tolist()
    #     predicted_label = df['predicted'].tolist()
    #     predicted_score = [score_curr[predicted_label[idx_curr]] for idx_curr, score_curr in enumerate(predicted_score_ori)]
    #     fpr, tpr, thresholds = metrics.roc_curve(target_label, predicted_score, pos_label='Yes')
    #     auc_curr = metrics.auc(fpr, tpr)
    #     results_dic[name]['auc'] = auc_curr

    return results_dic

def get_auc(test_bin):
    #print(test_bin)
    all_grps = test_bin.groupby('category')
    auc_dic = {}
    for name, df in all_grps:
        target_label = df['target_text'].tolist()
        predicted_score_ori = df['predicted_score'].tolist()
        predicted_label = df['predicted'].tolist()
        predicted_score = [score_curr["Yes"] for idx_curr, score_curr in enumerate(predicted_score_ori)] #predicted_label[idx_curr][0]
        # print(target_label)
        # print(predicted_score)
        fpr, tpr, thresholds = metrics.roc_curve(target_label, predicted_score, pos_label='Yes')
        auc_curr = metrics.auc(fpr, tpr)
        auc_dic[name] = {'auc': auc_curr}

    return auc_dic

def get_f1(precision, recall):
    return 2*((precision*recall)/(precision+recall))

def main(args):

    path_model = args.model_path
    test_df = pd.read_json(args.test_df, orient='columns')
    #test_df = test_df.head(30)
    #print(test_df)

    model = load_model(path_model)
    ## Might need to change this line depending on what classifier you are using

    test_df['category'] = test_df.source_text.apply(lambda x: x.split('Category - ')[1].split(',')[0])
    pred_df = get_predicted_df(test_df, model)
    auc_df = pd.DataFrame(get_auc(pred_df)).T
    results_df = pd.DataFrame(get_results(pred_df)).T
    result_all_df = pd.concat([results_df, auc_df], axis=1)
    # print(auc_df)
    # print(result_all_df)

    result_all_df.to_json(args.output_path)


def get_args():
    parser = argparse.ArgumentParser(description="Parser for the generating statistics with T5 models")
    parser.add_argument('--test_df', '-t', type=str, help="Path to Test Dataframe", required=True)
    parser.add_argument('--model_path', '-m', type=str, help="Path for the trained model", required=True)
    parser.add_argument('--output_path', '-o', type=str, help="Path to json file where results are stored", default="results.json")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    main(args)




