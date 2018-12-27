from types import ModuleType
from jinja2 import Template

import utils
import metrics
import keras_extension


def list_docs(module):
    methods = [getattr(module, i) for i in dir(module)
               if not i.startswith('_') and
               not isinstance(getattr(module, i), (ModuleType))]
    docs = dict()
    for m in methods:
        if hasattr(m, '__doc__'):
            doc = m.__doc__.split('\n')[0].strip()
        else:
            doc = 'No documentation found.'
        docs[m.__name__] = doc

    return docs


documentation = dict()

for m in [utils, metrics, keras_extension]:
    documentation.update(list_docs(m))

with open('./README.template', 'r') as f:
    template = Template(''.join(f.readlines()))

readme = template.render(documentation=documentation)

with open('./README.md', 'w') as f:
    f.write(readme)
