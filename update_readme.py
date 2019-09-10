from types import ModuleType
from jinja2 import Template
from collections import namedtuple

import utils
import metrics
# import keras_extension
import sklearn_extension


DOC = namedtuple('DOC', ['doc', 'file', 'lineno'])


def list_docs(module):
    module_name = module.__name__
    methods = [getattr(module, i) for i in dir(module)
               if not i.startswith('_') and
                 not isinstance(getattr(module, i), (ModuleType))]
    docs = dict()
    for m in methods:
        # unwrap the decorators
        while hasattr(m, '__wrapped__'):
            m = getattr(m, '__wrapped__')

        name = m.__name__
        
        # get documentation
        if hasattr(m, '__doc__') and m.__doc__ is not None:
            doc = m.__doc__.split('\n')[0].strip()
        else:
            doc = 'No documentation found.'

        try:
            if hasattr(m, '__code__'):
                # function object
                code = m.__code__
                file = code.co_filename
                # get the path starting from the module name
                file = file[file.index(module_name):].replace("\\", "/")
                lineno = code.co_firstlineno
            else:
                # class object
                file = '{}/__init__.py'.format(module_name)
                lineno = 1
        except:
            continue

        docs[module_name + '.' + name] = DOC(doc, file, lineno)

    return docs


documentation = dict()

for m in [utils, metrics, sklearn_extension]:
    documentation.update(list_docs(m))

with open('./README.template', 'r') as f:
    template = Template(''.join(f.readlines()))

readme = template.render(documentation=documentation)

with open('./README.md', 'w') as f:
    f.write(readme)
