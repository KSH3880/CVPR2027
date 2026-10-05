"""Run TensorBoard with protobuf JSON keyword compatibility, without package edits."""
import functools
import inspect

from google.protobuf import json_format


def patch_json_format():
    for name in ('MessageToJson', 'MessageToDict'):
        original = getattr(json_format, name)
        parameters = inspect.signature(original).parameters
        if ('including_default_value_fields' in parameters
                or 'always_print_fields_with_no_presence' not in parameters):
            continue

        def make_wrapper(function):
            @functools.wraps(function)
            def compatible(*args, **kwargs):
                if 'including_default_value_fields' in kwargs:
                    legacy_value = kwargs.pop('including_default_value_fields')
                    kwargs.setdefault('always_print_fields_with_no_presence', legacy_value)
                return function(*args, **kwargs)
            return compatible

        setattr(json_format, name, make_wrapper(original))


if __name__ == '__main__':
    patch_json_format()
    from tensorboard.main import run_main
    run_main()
