import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

submission = json.load(open("statistics/split_results_ovr.json"))
phase = submission["phase"]
if phase == "dev":
    ref_file = "scoring/dev.json"
elif phase == "test":
    ref_file = "test.json"
else:
    print(f"Phase {phase} is invalid.")
refs = json.load(open(ref_file))

assert submission["phase"] == refs["phase"]
assert submission["results"].keys() == refs["results"].keys()

submission = list(submission["results"].values())
refs = list(refs["results"].values())

labels = ["not-sarcasm", "multi-sarcasm", "text-sarcasm", "image-sarcasm"]

precision = precision_score(refs, submission, average="macro")
precision_per_label = precision_score(refs, submission, labels=labels, average=None)

recall = recall_score(refs, submission, average="macro")
recall_per_label = recall_score(refs, submission, labels=labels, average=None)

f1 = f1_score(refs, submission, average="macro")
f1_per_label = f1_score(refs, submission, labels=labels, average=None)

# Compute confusion matrix
cm = confusion_matrix(refs, submission, labels=labels)

# Extract TP, FP, FN, TN for each label
tp = cm.diagonal()
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
tn = cm.sum() - (fp + fn + tp)

scores = {
    'precision': precision,
    "recall": recall,
    "f1": f1
}

scores_per_label = {
    label: {
        'precision': precision_per_label[i]/4,
        'recall': recall_per_label[i]/4,
        'tp': int(tp[i]),
        'fp': int(fp[i]),
        'fn': int(fn[i]),
        'tn': int(tn[i])
    } for i, label in enumerate(labels)
}

with open('scoring/scores.json', 'w') as score_file:
    score_file.write(json.dumps(scores, indent=4))

with open('scoring/scores_per_label.json', 'w') as score_file:
    score_file.write(json.dumps(scores_per_label, indent=4))