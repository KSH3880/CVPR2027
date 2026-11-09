"""Launch TensorBoard 2.14 with the installed protobuf JSON API."""
import functools
import inspect
from google.protobuf import json_format


def adapt_json_api(function):
    if 'including_default_value_fields' in inspect.signature(function).parameters:
        return function

    @functools.wraps(function)
    def compatible(*args, **kwargs):
        if 'including_default_value_fields' in kwargs:
            kwargs['always_print_fields_with_no_presence'] = kwargs.pop('including_default_value_fields')
        return function(*args, **kwargs)
    return compatible


if __name__ == '__main__':
    json_format.MessageToJson = adapt_json_api(json_format.MessageToJson)
    json_format.MessageToDict = adapt_json_api(json_format.MessageToDict)
    from tensorboard.main import run_main
    run_main()
