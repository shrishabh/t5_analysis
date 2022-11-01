"""
Example script to show the usage of T5Wrapper to get the logit probabilities, given some labels from T5. Note that this is an
ad-hoc mechanism designed to work only with simple labels (labels that have single tokens). In case the label
is split into multiple tokens by the tokenizer - we compute the probabilities with the first token only.
"""

from T5Wrapper import T5Wrapper

if __name__ == '__main__':
    sample_text = "sst2 sentence: it confirms fincher ’s status " \
                  "as a film maker who artfully bends technical know-how to the service of psychological insight"
    labels = ['positive', 'negative']
    model = T5Wrapper()
    model.from_pretrained()  # loads a pre-tained t5-small model on GPU
    prediction = model.predict(source_text=sample_text, labels=labels)
    # prediction is a tuple, consisting of a list of prediction and the dictionary corresponding to softmax scores for
    # the labels.
    print(prediction)
