# How many exemplars can a cloud collect before it starts replacing old
# exemplars?
max_cloud_size = 13
# When an agent categorizes an incoming exemplar, what is the probability that
# the agent knows the correct category by ESP (= contextual information)?
probability_of_esp = .2
# When an agent produces a new exemplar, what is the probability that a given
# segment will be altered to match other segments in the same position in the
# cloud of its wordform?
probability_of_analogy = .7
# When an agent produces a new exemplar, what is the probability that a given
# segment will be altered to match other segments in the same position in the
# cloud of the wordform with the same lemma but the opposite case, relative to
# the probability of analogy within the wordform's own cloud?
paradigm_weight = .2
# When an agent produces a new exemplar, what is the probability that a given
# segment will be altered to match all other segments of the same type (in any
# cloud)?
probability_of_feat_analogy = .5
# When an agent produces a new exemplar, what is the probability that noise will
# be applied to the production of a given segment?
probability_of_noise = .2
# What is the standard deviation of the noise added to a feature, as a
# proportion of the range of that feature?
noise_sd_prop = .1
# When an agent produces a new exemplar, what is the probability that a final
# consonant will (partially) devoice?
probability_of_bias = .6
# What is the resolution with which maxima are identified during kernel density
# estimation?
kde_resolution = .1
# What is the bandwidth used during kernel density estimation?
kde_bandwidth = 5

# How many iterations of the simulation should be run?
iterations = 3000
# How many lemmas are there in the simulation?
num_lemmas = 2

# Should paradigm uniformity be applied?
paradigm_setting = True
# Should final segments be biased towards voicelessness?  Should medial segments
# be biased towards voicedness?
bias_setting = {'final': True, 'medial': False}
# What method should be used to categorize incoming exemplars?  'similarity':
# choose the category whose exemplars have the greatest average similarity to
# the incoming exemplar.  'bayes': do naive Bayesian classification.
categorization_setting = 'similarity'
# How should the informativity of a case in the paradigm be measured?
# 'entropy': the entropy of wordforms across the cloud (with binning of
# continuous features).  'classification': the performance of the categorization
# algorithm set above in `categorization_setting`.  'none': none.
informativity_setting = 'classification'
# Should paradigms have a unique base ('winner-take-all' application of
# `informativity_setting`)?
unique_base_setting = True
# Verbose output?
verbose_setting = False
# What is the largest amount a feature is allowed to change during entrenchment
# within a wordform's own cloud?
self_max_movement = 100
# During entrenchment within a wordform's own cloud, should a global maximum be
# returned (as opposed to a local maximum)?
self_top_value = True
# What is the largest amount a feature is allowed to change during entrenchment
# with the other member of the paradigm?
paradigm_max_movement = 100
# During entrenchment with the other member of the paradigm, should a global
# maximum be returned (as opposed to a local maximum)?
paradigm_top_value = True
# What is the largest amount a feature is allowed to change during entrenchment
# with all segments of the same type?
segment_max_movement = 100
# During entrenchment with all segments of the same type, should a global
# maximum be returned (as opposed to a local maximum)?
segment_top_value = False
