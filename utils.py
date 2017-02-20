from segment import Segment
from segment import all_features, feature_type, get_common_values
from copy import deepcopy
from random import choice
from math import floor, copysign, log
from statistics import mean
from NaiveBayes import NaiveBayes
from parameters import *

def cloud_form(cloud):
    """Return a single surface form that represents the entire cloud."""
    # Start off with a WordForm; make its segments empty.  All exemplars in the
    # cloud should have the same lemma and case; use the first one.
    surface = deepcopy(cloud[0])
    for i, seg in enumerate(surface.segments):
        if i < 3:
            seg.features = {}
    # Iterate through the segments of the exemplar.
    for i, seg in enumerate(surface.segments):
        if i < 3:
            # Iterate through each feature in the Segment.
            for feature in seg.possible_features():
                # Find the most common values of the feature in the cloud
                # provided at the relevant position.  Make sure these values are
                # compatible with the other features of the Segment.
                value_options = get_common_values(cloud, feature, i,
                                seg.contingent_possible_values(feature))
                # If there's at least one possible value, randomly select one of
                # the possible values and set it as the new value of the
                # feature.  Make sure the feature doesn't go outside the
                # permitted range.
                if len(value_options) > 0:
                    seg.features[feature] = choice(list(value_options))
                    seg.enforce_range(feature)
    # Return a string representation of the WordForm.
    return str(surface)

def entropy(feature, values):
    """Return the scaled entropy of the list of feature values provided."""
    # If the list is empty, return 0.
    if len(values) == 0:
        return 0
    # Otherwise, we can calculate the entropy.
    else:
        num_vals = len(all_features[feature]['values'])
        # If the values are continuous, bin them.
        if isinstance(values[0], (int, float)):
            frange = all_features[feature]['range']
            num_vals = 2
            bin_size = (frange[-1] - frange[0]) / num_vals
            values = [floor((val - frange[0]) / bin_size)
                      for val in values]
            values = [str(val)
                      if val < num_vals
                      else str(val - 1)
                      for val in values]
        # Calculate the entropy.
        entropy = copysign(sum(map(lambda p: p * log(p),
                                   [values.count(val) / len(values)
                                    for val in set(values)])), -1)
        max_entropy = copysign(log(1 / num_vals), -1)
        if max_entropy == 0:
            return 0
        else:
            return entropy / max_entropy

def predict_lemma(wordform, cloud, positions = None, features = None,
                  method = 'bayes'):
    """Predict the lemma of the WordForm based on the data in the cloud."""
    # Choose the most likely lemma based on the calculated similarities between
    # the WordForm and exemplars in the cloud.
    if method == 'similarity':
        lemma_sims = dict()
        # Iterate through all lemmas.
        candidate_lemmas = list({e.lemma for e in cloud})
        for lemma in candidate_lemmas:
            # Collect all exemplars of the lemma and their similarity to
            # the WordForm.
            lemma_sims[lemma] = mean([wf.similarity(wordform,
                                                    positions = positions,
                                                    features = features)
                                      for wf in cloud
                                      if wf.lemma == lemma
                                         and wf.case == wordform.case])
        # Return the lemma most similar to the WordForm.  If there's a
        # tie, pick among the tied winners at random.
        max_sim = max(lemma_sims[l] for l in lemma_sims)
        decision = choice([l for l in lemma_sims if lemma_sims[l] == max_sim])
        return decision
    # Do naive Bayesian classification.
    elif method == 'bayes':
        model = NaiveBayes()
        for e in cloud:
            # Collect the attributes and label of the exemplar, and add the
            # result to the model's training data.
            training = {'attributes': {'s' + str(i) + f: seg.features[f]
                                       for i, seg in enumerate(e.segments)
                                       if i < 3
                                          and (positions == None
                                               or i in positions)
                                       for f in seg.features
                                       if features is None or f in features},
                        'label': e.lemma,
                        'cases': 1}
            model.add_instances(training)
        # Tell the model that values of continuous features should be
        # interpreted as real numbers.
        for f in model.attributes:
            if feature_type(f[2:]) == 'continuous':
                model.set_real([f])
        # Train the model.
        model.train()
        # Deal with standard deviations that are zero - this can happen when
        # the cloud is initially seeded with a bunch of identical exemplars.  If
        # we don't do this, there's a divide-by-zero error.
        for stat in model.model['real_stat']:
            for lem in model.model['real_stat'][stat]:
                if model.model['real_stat'][stat][lem]['sigma'] == 0:
                    model.model['real_stat'][stat][lem]['sigma'] = .001
        # Collect the attributes of the WordForm to be categorized.
        wf_data = {'attributes': {'s' + str(i) + f: seg.features[f]
                                  for i, seg in enumerate(wordform.segments)
                                  if i < 3
                                  for f in seg.features
                                  if features is None or f in features}}
        # Predict the lemma of the WordForm to be categorized and return it.
        return model.preferredLabel(model.predict(wf_data))

def performance(cloud, positions = None, features = None, method = 'bayes'):
    """Return the performance of the specified classifier on the given data."""
    feature_results = []
    # Do leave-one-out cross-validation of the classification algorithm.
    for i, e in enumerate(cloud):
        comp_cloud = [exemplar
                      for j, exemplar in enumerate(cloud)
                      if not i == j]
        # Get the performance of the algorithm when only the specified features
        # are included.
        feature_results.append((e.lemma, predict_lemma(e, comp_cloud,
                                                       positions = positions,
                                                       features = features,
                                                       method = method)))
    # Return the performance of the feature-restricted classifier.
    feature_correct = len([fr for fr in feature_results if fr[0] == fr[1]])
    if len(feature_results) == 0:
        return 0
    else:
        return feature_correct / len(feature_results)
