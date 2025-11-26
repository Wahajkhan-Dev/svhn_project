"""Evaluate saved model: confusion, mis_idx, classification report."""
import argparse, os, json, numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def main(out):
    with open(f'{out}/preds.json','r') as f:
        preds = json.load(f)
    y_pred = np.array(preds['pred'])
    y_true = np.array(preds['true'])
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion matrix:\n', cm)
    print('\nClassification report:\n', classification_report(y_true, y_pred, digits=3))
    mis = (y_pred != y_true).nonzero()[0].tolist()
    with open(f'{out}/mis_idx.json','w') as f:
        json.dump(mis, f)
    print('Saved mis_idx.json to', out)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='outputs')
    args = p.parse_args()
    main(args.out)
