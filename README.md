# Exemplar-Positional-Neutralization
An agent-based exemplar model of positional neutralization in morphological paradigms

![Sample simulation run](https://github.com/kaplanas/Exemplar-Positional-Neutralization/blob/master/sim_illustration.png)

## Author
Abby Kaplan ([kaplanas](https://github.com/kaplanas))

## About
This code runs an agent-based exemplar simulation of words that participate in morphological paradigms.  [This manuscript](https://github.com/kaplanas/Exemplar-Positional-Neutralization/blob/master/exemplar_positional_neutralization.pdf) describes a series of simulations that explore whether an exemplar model can replicate German-style final devoicing.  (A related earlier project is reported [here](http://journals.linguisticsociety.org/proceedings/index.php/amphonology/article/view/3745).)

The code consists of

+ Python code that runs the simulation and dumps the results to a JSON file, and  
+ R code that reads the JSON and plots the results.

A simulation contains two agents and a small lexicon.  For each word in the lexicon, each agent has a "cloud" of remembered exemplars of that word.  Agents communicate by producing, perceiving, and storing exemplars.  A single interaction between the agents proceeds as follows:

1. Production (Agent 1)  
  a. Choose a wordform at random.  
  b. Choose an exemplar at random from the wordform's cloud; copy it into a new wordform for production.  
  c. Modify the production.  
    i. Entrenchment: make the production more similar to other exemplars.  
    ii. Phonetic bias: apply pressure towards final devoicing.  
    iii. Noise.  
2. Perception (Agent 2)  
  a. Determine the wordform that the wordform most likely came from.  
  b. Store the production in the appropriate cloud, replacing a randomly selected old exemplar.  
3. Return to step 1 and switch roles.

## Dependencies
+ `nltk`
+ `numpy`
+ `pyqt_fit`
+ `NaiveBayes`

## Usage
To run the code,

1. Download the .py and .R files and put them in the same directory.  
2. Edit `parameters.py` so that it has the settings you want.  
3. (You may also want to edit the `initialize_agent()` function in `simulation.py`, if you want to change the number/shape/composition of the lemmas.)  
4. Run `simulation.py`.  
5. Edit `plot_results.R` so that `in_file_name` is the name of the JSON file you just created and `path` is the name of the file to which you want to write the graph.  
6. Run `plot_results.R`.
