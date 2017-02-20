import json, datetime
from agent import Agent
from wordform import WordForm
from segment import Segment

log_file_name = 'sim_raw_' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M.%S") + '.json'

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        """Deal with custom classes for JSON."""
        # Turn Agents, WordForms, and Segments into dictionaries.
        if isinstance(obj, (Agent, WordForm, Segment)):
            return {key: CustomEncoder.default(self, obj.__dict__[key])
                    for key in obj.__dict__}
        # Round floats to one decimal place.
        elif isinstance(obj, float):
            return round(obj, 1)
        # Call to_json recursively on lists and dictionaries.
        elif isinstance(obj, list):
            return [CustomEncoder.default(self, p) for p in obj]
        elif isinstance(obj, dict):
            return {key: CustomEncoder.default(self, obj[key]) for key in obj}
        # Deal with things that seem to make JSON unhappy.
        elif isinstance(obj, (int, str)):
            return obj
        # Otherwise, use JSON's default.
        else:
            return json.JSONEncoder.default(self, obj)

def start_logging():
    """Print a header in the log file."""
    with open(log_file_name, 'w') as log_file:
        log_file.write('HEADER')
        log_file.write()

def log_state(agent):
    """Record the Agent's current state to the log file."""
    with open(log_file_name, 'a') as log_file:
        log_file.write(json.dumps(agent, cls = CustomEncoder))
        log_file.write('\n')
