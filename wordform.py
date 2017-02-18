from segment import Segment
from segment import all_features, feature_distance, get_common_values
from segment import feature_type
from utils import entropy, performance
from copy import deepcopy
from random import uniform, choice
from numpy.random import choice as wchoice
from parameters import *
from sim_type_parameters import *

cases = {'abs': {'name': 'Absolutive',
                 'suffix': ''},
         'erg': {'name': 'Ergative',
                 'suffix': 'i'}}

class WordForm:
    """A class for wordforms (strings of Cs and Vs)"""
    
    def __init__(self, segments, lemma = None, case = None):
        """Initalize with a user-supplied list of segments."""
        self.segments = segments
        if isinstance(self.segments, str):
            self.segments = [Segment.new_segment(s) for s in self.segments]
        self.lemma = lemma
        self.case = case
    
    @classmethod
    def random_segs(cls, shape, lemma = None, case = None):
        """Create a WordForm of the given CV shape with random segments."""
        # For each C or V segment in `shape`, initialize a random Segment of the
        # appropriate type.  Initialize a new WordForm with all these Segments.
        return cls([Segment(seg_type = seg) for seg in shape], lemma, case)
    
    def add_suffix(self, suffix):
        """Add the suffix vowel."""
        # Append the suffix vowel to this WordForm.
        self.segments.append(Segment.new_segment(suffix))
    
    def entrench(self, cloud, paradigms, informativity, categorization,
                 unique_base):
        """Move the WordForm closer to the middle of various clouds."""
        self.entrench_word(cloud, paradigms, informativity, categorization,
                           unique_base)
        self.entrench_segments(cloud)
    
    def entrench_word(self, cloud, paradigms, informativity, categorization,
                      unique_base):
        """Entrench at the level of the WordForm."""
        # Entrench within the WordForm's own cloud.  Iterate over positions in
        # the WordForm (up to three Segments).
        for pos, seg in enumerate(self.segments):
            if pos < 3:
                # Iterate over features.
                for feat in seg.features:
                    if uniform(0, 1) < probability_of_analogy:
                        # Collect other values of the feature across the cloud.
                        # Since this is the WordForm's own cloud, set all the
                        # weights to 1.
                        wv = [(e.segments[pos].features[feat], 1)
                              for e in cloud
                              if e.lemma == self.lemma
                                 and e.case == self.case]
                        # Entrench the segment based on these values.
                        seg.entrench_feature(feat, wv,
                                             top_value = self_top_value,
                                             max_movement = self_max_movement)
        # Entrench within other clouds of the same paradigm.
        if paradigms:
            # Iterate over positions in the WordForm (up to three Segments).
            for pos, seg in enumerate(self.segments):
                if pos < 3:
                    # Iterate over features.
                    for feat in seg.features:
                        if uniform(0, 1) < (probability_of_analogy *
                                            paradigm_weight):
                            # Get the weight for each case.
                            weights = dict()
                            # If informativity is measured via the entropy
                            # method, the weight of a case is proportional to
                            # the entropy of the feature across all lemmas of
                            # that case.
                            if informativity == 'entropy':
                                weights = {c: entropy(feat, [e.segments[pos].\
                                                             features[feat]
                                                             for e in cloud
                                                             if e.case == c])
                                           for c in cases}
                            # If informativity is measured via a classification
                            # algorithm, the weight of a case is proportional to
                            # the performance of the classifier on lemmas within
                            # that case using just the current feature.
                            elif informativity == 'classification':
                                weights = {c: performance([e
                                                           for e in cloud
                                                           if e.case == c],
                                                          positions = [pos],
                                                          features = [feat],
                                                        method = categorization)
                                           for c in cases}
                            # If informativity is not measured, set the weights
                            # of all cases to 1.
                            elif informativity == 'none':
                                weights = {c: 1
                                           for c in cases}
                            # If paradigms are required to have a unique base,
                            # the winner takes all the weight.
                            if unique_base:
                                max_weight = max(weights.values())
                                for c in weights:
                                    if weights[c] < max_weight:
                                        weights[c] = 0
                            # Collect other values of the feature across the
                            # cloud, and pair them with their weights.
                            wv = [(e.segments[pos].features[feat],
                                   weights[e.case])
                                  for e in cloud
                                  if e.lemma == self.lemma
                                     and e.case != self.case]
                            # Entrench the segment based on these values.
                            seg.entrench_feature(feat, wv,
                                                 top_value = paradigm_top_value,
                                           max_movement = paradigm_max_movement)
    
    def entrench_segments(self, cloud):
        """Entrench at the level of the Segment."""
        # Iterate over features.
        for feat in all_features:
            if feature_type(feat) == 'continuous':
                # Collect all values of the feature across the cloud.
                values = [(s.features[feat], 1)
                          for e in cloud
                          for p, s in enumerate(e.segments)
                          if p < 3 and feat in s.features]
                # Iterate over Segments.
                for pos, seg in enumerate(self.segments):
                    if pos < 3:
                        if uniform(0, 1) < probability_of_feat_analogy:
                            # Entrench the feature of the Segment based on
                            # values across the cloud.
                            seg.entrench_feature(feat, values,
                                                 top_value = segment_top_value,
                                            max_movement = segment_max_movement)
    
    def add_noise(self):
        """Add noise to the non-suffix segments in the WordForm."""
        self.segments = deepcopy(self.segments)
        # Iterate through each of the first three Segments in the WordForm.
        for i in range(3):
            # Add noise to each Segment.
            self.segments[i].add_noise()
    
    def add_bias(self, bias_types):
        """Add articulatory bias to the non-suffix segments in the WordForm."""
        self.segments = deepcopy(self.segments)
        # Add the biases specified in the argument.
        if bias_types['final'] and len(self.segments) == 3:
            self.segments[2].add_bias('voiceless')
        if bias_types['medial'] and len(self.segments) == 4:
            self.segments[2].add_bias('voiced')
    
    def distance(self, wf, positions = None, features = None):
        """Return the distance between this WordForm and the one provided."""
        dist = 0
        # Iterate through positions in the WordForm.
        for position in range(3):
            if positions == None or position in positions:
                if self.segments[position].seg_type == wf.segments[position].\
                                                       seg_type:
                    # Use the WordForm's Segment to determine the possible
                    # features of Segments in this position.
                    for feature in self.segments[position].possible_features():
                        if features == None or feature in features:
                            my_value = self.segments[position].\
                                       get_feature_value(feature)
                            comp_value = wf.segments[position].\
                                         get_feature_value(feature)
                            dist += abs(feature_distance(feature, my_value,
                                                         comp_value))
                else:
                    return 100
        return dist
    
    def similarity(self, wf, positions = None, features = None):
        """Return the similarity between this WordForm and the one provided."""
        # The similarity is the inverse square of the distance between the two
        # WordForms.  Impose a minimum on distances (to deal with zero).
        dist = self.distance(wf, positions = positions, features = features)
        if dist < .1:
            dist = .1
        sim = 1 / (dist ** 2)
        return sim
    
    def sr(self):
        """Return just the surface representation of the WordForm."""
        return ''.join([str(seg) for seg in self.segments])
    
    def __len__(self):
        """Return the number of segments in this WordForm."""
        return len(self.segments)
    
    def __repr__(self):
        r = 'WordForm(' + ', '.join([repr(seg) for seg in self.segments])
        if not self.lemma is None:
            r = r + ', lemma = ' + str(self.lemma)
        if not self.case is None:
            r = r + ', case = ' + str(self.case)
        r = r + ')'
        return r
    
    def __str__(self):
        s = ''.join([str(seg) for seg in self.segments])
        if not self.lemma is None:
            s = s + '-' + str(self.lemma)
        if not self.case is None:
            s = s + '-' + str(self.case)
        return s
