import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from carbongrid.inference.provider import InferenceProvider
from unittest.mock import MagicMock

provider = InferenceProvider(max_concurrent_inferences=1)
print('Before mock:', provider.ensure_loaded)

# Try to mock it
provider.ensure_loaded = lambda *a, **kw: ('model', 'tokenizer')
print('After mock:', provider.ensure_loaded)

result = provider.ensure_loaded('fp16')
print('Result:', result)