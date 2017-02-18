from random import choice, uniform, gauss
from statistics import mean
from nltk import FreqDist
from math import copysign
from numpy.random import choice as wchoice
from numpy import arange
from pyqt_fit import kde
from parameters import *
from sim_type_parameters import *
 
# I really should have made a class for features.  Alas, inertia has won.
# For each feature, specify:
#   - The type of segment for which the feature is appropriate (C or V).
#   - The possible categorical values of the feature.
#       * If it's a categorical feature, the values are specified in a set.
#       * If it's a continuous feature, the values are specified in a list.
#         This list specifies, in order, the category labels that span the range
#         of continuous values.
#   - The range of numeric values for a continuous feature.
#       * The first number in the tuple gives the smallest possible value; the
#         last number gives the largest possible value.
#       * Intermediate numbers give the boundaries between category labels.
all_features = {'place': {'type': 'C',
                          'values': {'lab'}},
                'vot': {'type': 'C',
                        'values': ['voiced', 'voiceless'],
                        'range': (0, 50, 100)},
                'height': {'type': 'V',
                           'values': {'high'}},
                'backness': {'type': 'V',
                             'values': {'front'}}}
# For each segment, specify:
#   - The type of segment (C or V).
#   - The features of the segment and their values.
all_segments = {'p': {'type': 'C',
                      'features': {'place': 'lab', 'vot': 'voiceless'}},
                'b': {'type': 'C',
                      'features': {'place': 'lab', 'vot': 'voiced'}},
                'i': {'type': 'V',
                      'features': {'height': 'high', 'backness': 'front'}}}

def feature_type(feature):
    """Determine whether the feature is categorical or continuous."""
    if feature in all_features:
        # If the possible values of a feature are given in a set, it's
        # categorical.
        if isinstance(all_features[feature]['values'], set):
            return 'categorical'
        # If the possible values of a feature are given in a list, it's
        # continuous.
        elif isinstance(all_features[feature]['values'], list):
            return 'continuous'
        # Otherwise, we have a problem.
        else:
            raise CategoryNotDefinedError(feature)
    else:
        raise FeatureNotFoundError(feature)

def feature_range(feature):
    """Return the range of a continuous feature."""
    ftype = feature_type(feature)
    if ftype == 'continuous':
        return all_features[feature]['range']

def num_categories(feature):
    """Return the number of categories for the given feature."""
    if feature in all_features:
        # The number of categories is the number of possibilities listed in the
        # feature specification.
        return len(all_features[feature]['values'])
    else:
        raise FeatureNotFoundError(feature)

def category_to_range(feature, category):
    """Return the range of values of a category of a continuous feature."""
    if feature_type(feature) == 'continuous':
        # Get the linear position of the category label given in the argument.
        category_index = all_features[feature]['values'].index(category)
        # The range for the category in that position is specified in that
        # feature's range specification.
        range_min = all_features[feature]['range'][category_index]
        range_max = all_features[feature]['range'][category_index + 1]
        return (range_min, range_max)

def value_to_category(feature, value):
    """Return the category of a value of a continuous feature."""
    if feature_type(feature) == 'continuous' and not isinstance(value, str):
        # Check whether the value is too small for the feature.
        if all_features[feature]['range'][0] > value:
            raise ValueTooSmallError(feature, str(value))
        # Iterate through category labels until you find the one whose range
        # covers the value provided in the argument.
        for (i, cat) in enumerate(all_features[feature]['values']):
            if all_features[feature]['range'][i + 1] >= value:
                return cat
        # If you still haven't found a matching category, the value is too big
        # for the feature.
        raise ValueTooBigError(feature, str(value))
    else:
        return value

def features_compatible_with_segment(features, segment):
    """Check whether the features given are compatible with some segment."""
    if segment in all_segments:
        # If any feature-value pair in the dictionary provided is incompatible
        # with the segment provided, the two are not compatible.
        for feature in features:
            # Use categorical labels for continuous features.
            cat_value = value_to_category(feature, features[feature])
            # If the feature isn't listed as one of the segment's feature, the
            # two are not compatible.
            if not feature in all_segments[segment]['features']:
                return False
            # If the feature has a different value from the one listed with the
            # segment, the two are not compatible.
            if not cat_value == all_segments[segment]['features'][feature]:
                return False
        # Otherwise, the two are compatible.
        return True
    else:
        raise SegmentNotFoundError(segment)

def possible_feature_combination(features):
    """Check whether any segment has the combination of features given."""
    # If there's any segment that's compatible with the specified features, that
    # feature combination is possible.
    for segment in all_segments:
        if features_compatible_with_segment(features, segment):
            return True
    # Otherwise, that feature combination is impossible.
    return False

def get_all_values(cloud, feature, position = None, possible_values = None):
    """"Return all values of the feature (at some position) in the cloud."""
    if feature in all_features:
        # If the user specified the possible values of the feature, restrict the
        # search to just those values.  Otherwise, consider all possible values
        # (and note this fact).
        restricted = True
        possibilities = possible_values
        if possibilities is None:
            possibilities = all_features[feature]['values']
            restricted = False
        # If no values of the feature are allowed, give up.
        if len(possibilities) == 0:
            return None
        # Collect all the values of the feature, restricted by position if the
        # user has specified this.
        if position == None:
            value_list = [s.features[feature]
                          for e in cloud
                          for s in e.segments]
        else:
            value_list = [e.segments[position].features[feature]
                          for e in cloud]
        # If possible feature values are restricted, narrow down the list
        # accordingly.
        f_type = feature_type(feature)
        if f_type == 'categorical':
            if restricted:
                value_list = [v for v in value_list
                              if v in possibilities]
        elif f_type == 'continuous':
            if restricted:
                value_list = [v for v in value_list
                              if value_to_category(feature, v) in possibilities]
        return value_list
    else:
        raise FeatureNotFoundError(feature)

def get_common_values(cloud, feature, position = None, possible_values = None):
    """Return the most common values of the feature in the cloud."""
    if feature in all_features:
        # Get all the values of the specified feature at the specified position
        # in the cloud provided with the restrictions provided.
        value_list = get_all_values(cloud, feature, position, possible_values)
        f_type = feature_type(feature)
        # Get common values for categorical features.
        if f_type == 'categorical':
            # Find all values that are tied for the most frequent and return
            # them.
            value_counts = FreqDist(value_list)
            top_values = {v for v in value_counts
                          if value_counts[v] ==\
                          max(value_counts.values())}
            return top_values
        # Get common values for continuous features.
        elif f_type == 'continuous':
            # Return the average of the feature values.
            if len(value_list) > 0:
                return {mean(value_list)}
            else:
                return set()
    else:
        raise FeatureNotFoundError(feature)

def feature_distance(feat, val1, val2):
    """Return the distance between the two feature values provided."""
    f_type = feature_type(feat)
    # For categorical features, return 0 if the features match and .7 otherwise.
    if f_type == 'categorical':
        if val1 == val2:
            return .7
        else:
            return 1
    # For continuous features, return the distance between the features,
    # normalized by the range of possible values for that feature.
    elif f_type == 'continuous':
        frange = all_features[feat]['range']
        dist = (val1 - val2) / (frange[-1] - frange[0])
        return dist

def density_maxima(feature, weighted_values):
    """Return the maxima of a KDE based on the values provided."""
    kde_est = kde.KDE1D([v for v, w in weighted_values],
                        weights = [w for v, w in weighted_values],
                        bandwidth = kde_bandwidth)
    xs = arange(all_features[feature]['range'][0],
                all_features[feature]['range'][-1],
                kde_resolution).tolist()
    ys = kde_est(xs).tolist()
    left_maxima = [a > b for a, b in zip(ys, [0] + ys[:-1])]
    right_maxima = [a > b for a, b in zip(ys, ys[1:] + [0])]
    maxima = [l & r for l, r in zip(left_maxima, right_maxima)]
    max_coords = [(x, y) for x, y, m in zip(xs, ys, maxima) if m]
    return max_coords

class Segment:
    """A class for consonants and vowels"""
    
    def __init__(self, seg_type = None, **kwargs):
        """Initialize, filling unspecified features with random values."""
        self.features = dict()
        # If the segment type is not specified, randomly choose either a
        # consonant or a vowel.
        self.seg_type = seg_type
        if self.seg_type is None:
            self.seg_type = choice(['C', 'V'])
        if not self.seg_type in ['C', 'V']:
            raise SegmentTypeNotDefinedError(self.seg_type)
        # For each feature, either initialize it to the value given in one of
        # the named arguments or initialize it to a random value.
        for feature in self.possible_features():
            f_type = feature_type(feature)
            # Initialize categorical features.
            if f_type == 'categorical':
                # If a value was provided in the argument, initialize the
                # feature to that value.
                if feature in kwargs:
                    self.features[feature] = kwargs[feature]
                # If no value was provided in the argument, initialize the
                # feature to a random value.
                else:
                    self.set_random_value(feature)
            # Initialize continuous features.
            elif f_type == 'continuous':
                if feature in kwargs:
                    # If a number was provided in the argument, initialize the
                    # feature to that value.
                    if isinstance(kwargs[feature], (int, float)):
                        self.features[feature] = kwargs[feature]
                    # If a string was provided in the argument, initialize the
                    # feature to a random value in the appropriate range.
                    elif isinstance(kwargs[feature], str):
                        self.features[feature] = uniform(\
                            *category_to_range(feature, kwargs[feature]))
                    else:
                        raise ValueTypeNotDefinedError(kwargs[feature])
                # If nothing was provided in the argument, initialize the
                # feature to a random value in the whole range of possible
                # values.
                else:
                    self.set_random_value(feature)
                    self.enforce_range(feature)
    
    @classmethod
    def new_segment(cls, segment):
        """Create a Segment based on its string representation."""
        # Determine whether the segment specified in the argument is a C or V.
        possible_types = {all_segments[seg]['type'] for seg in all_segments
                          if seg == segment}
        # If the type of segment can be uniquely determined, proceed.
        if len(possible_types) == 1:
            return Segment(seg_type = list(possible_types)[0],
                           **all_segments[segment]['features'])
        # If the segment could be either C or V, that's bad.
        elif len(possible_types) > 1:
            raise TooManySegmentsError(segment)
        # If the segment can't be either C or V, that's bad too.
        else:
            raise SegmentNotFoundError(segment)
    
    def possible_features(self):
        """Return possible features of the Segment, based on its type."""
        return {f for f in all_features
                if all_features[f]['type'] == self.seg_type}
    
    def get_feature_value(self, feature, convert_to_categorical = False):
        """Return value of specified feature for this Segment."""
        if feature in self.features:
            value = self.features[feature]
            # If the user specified a continuous feature and requested that it
            # be converted to a category label, do so.
            if (isinstance(value, int) or isinstance(value, float))\
               and convert_to_categorical:
                return value_to_category(feature, value)
            # Otherwise, just return the value of the feature.
            else:
                return value
        else:
            raise FeatureNotSpecifiedError(feature)
    
    def enforce_range(self, feature):
        """Adjust the value of the given feature if it's outside its range."""
        if feature in self.features:
            # If the feature is continuous, some adjustment might be necessary.
            if 'range' in all_features[feature]:
                # Get the category labels that this feature is allowed to fall
                # under.
                value_options = self.contingent_possible_values(feature)
                # Get the range for each possible category label and find the
                # overall minimum and maximum.
                feature_ranges = [category_to_range(feature, cat)
                                  for cat in value_options]
                feature_min = min(f_range[0] for f_range in feature_ranges)
                feature_max = max(f_range[1] for f_range in feature_ranges)
                # If the value is too small, set it to the minimum.
                if self.features[feature] < feature_min:
                    self.features[feature] = feature_min
                # If the value is too large, set it to the maximum.
                elif self.features[feature] > feature_max:
                    self.features[feature] = feature_max
        else:
            raise FeatureNotSpecifiedError(feature)
    
    def contingent_possible_values(self, feature):
        """Return possible values of a feature given rest of the Segment."""
        if feature in self.possible_features():
             # Get all the features of this Segment except the one specified by
             # the user.
             feature_combo = {f: self.features[f] for f in self.features
                              if not f == feature}
             # Get all segments that are compatible with this combination of
             # features.
             possible_segments = [all_segments[seg] for seg in all_segments
                                  if features_compatible_with_segment(\
                                      feature_combo, seg)
                                  and all_features[feature]['type'] ==\
                                      all_segments[seg]['type']]
             # Get all values of the feature of interest across these segments.
             # These are all possible values of the feature of interest, given
             # the other feature specifications in this Segment.
             other_features = {seg['features'][feature]
                               for seg in possible_segments}
             return other_features
        else:
            raise FeatureNotSpecifiedError(feature)
    
    def set_random_value(self, feature):
        """Set the specified feature to a random value."""
        if feature in self.possible_features():
            f_type = feature_type(feature)
            # Choose a random value for categorical features.  Make sure the
            # chosen value is compatible with other features of this Segment.
            if f_type == 'categorical':
                value_options = self.contingent_possible_values(feature)
                self.features[feature] = choice(list(value_options))
            # Choose a random value in the appropriate range for continuous
            # features.  Make sure the chosen value is compatible with other
            # features of this Segment.
            elif f_type == 'continuous':
                value_options = self.contingent_possible_values(feature)
                value_choice = choice(list(value_options))
                f_range = category_to_range(feature, value_choice)
                self.features[feature] = uniform(f_range[0], f_range[1])
        else:
            raise FeatureNotPossibleError(feature)
    
    def entrench_feature(self, feature, weighted_values, top_value,
                         max_movement):
        """Perturb a feature based on the values provided."""
        if all_features[feature]['type'] == self.seg_type:
            f_type = feature_type(feature)
            # Entrench categorical features.
            if f_type == 'categorical':
                # If the user specified that the top value is to be returned,
                # find the top value and return it.
                if top_value:
                    val_counts = {val: sum(w for v, w in weighted_values
                                             if v == val)
                                  for val in set([v
                                                  for v, w in weighted_values])}
                    max_count = max(val_counts.values())
                    top_vals = {val for val in val_counts
                                if val_counts[val] == max_count}
                    self.features[feature] = choice(list(top_vals))
                # Otherwise, choose a value at random, with more heavily
                # weighted values more likely to be chosen.
                else:
                    total_weight = sum(w for v, w in weighted_values)
                    if total_weight > 0:
                        self.features[feature] = wchoice([v for v, w
                                                            in weighted_values],
                                                          p = [w / total_weight
                                                               for v, w
                                                            in weighted_values])
            # Entrench continuous features.
            elif f_type == 'continuous':
                top_val = self.features[feature]
                total_weights = sum(w for v, w in weighted_values)
                if total_weights > 0:
                    # Get the local maxima of a KDE based on the collected
                    # values of the feature.
                    maxima = density_maxima(feature, weighted_values)
                    # If the user specified that the top value is to be used,
                    # find the global maximum.
                    if top_value:
                        global_maxes = [v for v, d in maxima
                                        if d == max(d for v, d in maxima)]
                        if len(global_maxes) > 0:
                            top_val = choice(global_maxes)
                    # Otherwise, use the nearest local maximum.
                    else:
                        nearest_maxes = [v for v, d in maxima
                                         if abs(v - self.features[feature]) ==
                                            min(abs(v - self.features[feature])
                                                for v, d in maxima)]
                        if len(nearest_maxes) > 0:
                            top_val = choice(nearest_maxes)
                    target = top_val
                    if abs(top_val - self.features[feature]) > max_movement:
                        target = self.features[feature] +\
                                 copysign(max_movement,
                                          top_val - self.features[feature])
                    self.features[feature] = target
                    self.enforce_range(feature)
    
    def add_noise(self):
        """Randomly perturb the features of the Segment."""
        # Iterate through each feature individually.
        for feature in self.features:
            if uniform(0, 1) < probability_of_noise:
                f_type = feature_type(feature)
                # If the feature is cateogrical, change to a random value with
                # the probability specified above.
                if f_type == 'categorical':
                    self.set_random_value(feature)
                # If the feature is continous, apply Gaussian noise.  The
                # standard deviation is one fifth of the size of the entire
                # range of possible values for the feature.
                elif f_type == 'continuous':
                    feature_max = all_features[feature]['range'][0]
                    feature_min = all_features[feature]['range'][-1]
                    sd = (feature_max - feature_min) * noise_sd_prop
                    self.features[feature] = gauss(self.features[feature], sd)
                    self.enforce_range(feature)
    
    def add_bias(self, bias_type):
        """Add articulatory biases to the Segment."""
        # Iterate through each feature individually.
        for feature in self.features:
            # Affect only voicing.
            if feature == 'vot' and uniform(0, 1) < probability_of_bias:
                if bias_type == 'voiced':
                    target = 25
                elif bias_type == 'voiceless':
                    target = 75
                change = (target - self.features[feature]) / 2
                self.features[feature] += change
                self.enforce_range(feature)
    
    def __repr__(self):
        rep_string = 'Segment(seg_type = ' + self.seg_type
        for feature in self.features:
            rep_string += ', ' + feature + ' = ' + str(self.features[feature])
        rep_string += ')'
        return rep_string
    
    def __str__(self):
        possible_segments = {seg for seg in all_segments
                             if features_compatible_with_segment(self.features,
                                                                 seg)}
        if len(possible_segments) == 1:
            return list(possible_segments)[0]
        else:
            return 'X'

class SegmentError(Exception):
    """Base class for exceptions in the `segment` module."""
    pass

class SegmentNotFoundError(SegmentError):
    """Exception raised when a segment can't be found in the master list."""
    pass

class TooManySegmentsError(SegmentError):
    """Exception raised when identical segments are found in the master list."""
    pass

class SegmentTypeNotDefinedError(SegmentError):
    """Exception raised when a segment type isn't defined."""
    pass

class FeatureError(SegmentError):
    """Exceptions raised for problems with features."""
    pass

class FeatureNotFoundError(FeatureError):
    """Exception raised when a feature can't be found in the master list."""
    pass

class FeatureNotSpecifiedError(FeatureError):
    """Exception raised when a feature is unexpectedly not specified."""
    pass

class FeatureNotSpecifiedError(FeatureError):
    """Exception raised when a feature is unexpectedly impossible."""
    pass

class CategoryNotDefinedError(FeatureError):
    """Exception raised when a feature's category can't be determined."""
    pass

class WrongTypeError(FeatureError):
    """Exception raised when a feature has the wrong type for its purpose."""
    pass

class ValueTypeNotDefinedError(FeatureError):
    """Exception raised when a feature value's type isn't defined."""
    pass

class ValueOutsideRangeError(FeatureError):
    """Exception raised when the value of a feature is outside its range."""
    pass

class ValueTooSmallError(ValueOutsideRangeError):
    """Exception raised when the value of a feature is too small."""
    pass

class ValueTooBigError(ValueOutsideRangeError):
    """Exception raised when the value of a feature is too big."""
    pass
