import argparse
from ast import literal_eval

# This is used to pass a variable number of parameter arguments as strings, subsequently loading them into a dict with the correct type
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            try:
                parsed_value = literal_eval(value)
            except (ValueError, SyntaxError):
                parsed_value = value
            getattr(namespace, self.dest)[key] = parsed_value
