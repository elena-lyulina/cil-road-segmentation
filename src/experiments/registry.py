# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Taken from https://github.com/Julienbeaulieu/kaggle-computer-vision-competition/blob/master/src/tools/registry.py
# with some additional functionality for error handling and imports

from os.path import relpath, basename

from src.constants import SRC_PATH, ROOT_PATH


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.
    Eg. creating a registry:
        some_registry = Registry({"default": default_module})
    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn

    # Added for better error handling
    def get(self, __key):
        if __key not in self:
            registered = [f"\"{k}\"" for k in list(self.keys())]
            registered = ", ".join(registered)
            raise ValueError(f"Implementation of \"{__key}\" is not registered, make sure you registered this "
                             f"implementation or that you are using the correct name\n"
                             f" Registered implementations: {registered}\n"
                             f"Additionally, check that the desired implementation is imported."
                             )
        return self[__key]


def import_files_from(dir: str):
    # A bit ugly but we need to import all implementations to be able to find them via registry
    pwd = SRC_PATH.joinpath(dir)
    for x in pwd.rglob('*.py'):
        module_name = relpath(x, ROOT_PATH)[:-3].replace('/', '.')
        if not basename(x).startswith('__'):
            __import__(module_name, globals(), locals())
