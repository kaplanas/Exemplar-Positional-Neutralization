from wordform import WordForm, cases
from agent import Agent
from segment import Segment
from log_utils import log_state
from random import gauss
from parameters import *

num_lemmas = 3
lemmas = range(1, num_lemmas + 1)

def initialize_agent(agent):
    """Seed an agent with initial exemplars."""
    agent.timestep = 0
    lemma_shapes = ['CVC', 'CVC']
    # For each case,
    for case in cases:
        # fill the cloud with initial exemplars for each lemma shape,
        wfs = [WordForm.random_segs(shape, lemma = li + 1, case = case)
               for (li, shape) in enumerate(lemma_shapes)
               for i in range(max_cloud_size)]
        # set VOTs for each consonant according to the desired initial state,
        for wf in wfs:
            for (pos, seg) in enumerate(wf.segments):
                if seg.seg_type == 'C':
                    if wf.lemma == 1:
                        seg.features['vot'] = gauss(25, 5)
                    elif wf.lemma == 2:
                        seg.features['vot'] = gauss(75, 5)
        # and add a suffix if thise case is supposed to have one.
        if len(cases[case]['suffix']) == 1:
            for wf in wfs:
                wf.add_suffix(cases[case]['suffix'])
        agent.add_exemplars(wfs)

# Initialize two agents, log them to a file, and print out their initial
# exemplars.
a1 = Agent(1)
initialize_agent(a1)
log_state(a1)
print()
print('*** AGENT 1 ***')
a1.print_exemplars()
a2 = Agent(2)
initialize_agent(a2)
log_state(a2)
print('*** AGENT 2 ***')
a2.print_exemplars()

# Run the simulation.
for i in range(1, iterations + 1):
    if i % 200 == 0:
        print(i)
    # Agent 1 produces a word and Agent 2 stores it.  Log what happened.
    a2.store(a1.produce(paradigms = paradigm_setting,
                        bias = bias_setting,
                        informativity = informativity_setting,
                        categorization = categorization_setting,
                        unique_base = unique_base_setting),
             prob_esp = probability_of_esp,
             categorization = categorization_setting)
    a2.timestep += 1
    log_state(a2)
    # Agent 2 produces a word and Agent 1 stores it.  Log what happened.
    a1.store(a2.produce(paradigms = paradigm_setting,
                        bias = bias_setting,
                        informativity = informativity_setting,
                        categorization = categorization_setting,
                        unique_base = unique_base_setting),
             prob_esp = probability_of_esp,
             categorization = categorization_setting)
    a1.timestep += 1
    log_state(a1)

print()
print('*** AGENT 1 ***')
a1.print_exemplars()
print('*** AGENT 2 ***')
a2.print_exemplars()
