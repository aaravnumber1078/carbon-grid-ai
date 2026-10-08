import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from carbongrid.inference.provider import InferenceProvider
from unittest.mock import MagicMock
import threading
import time

provider = InferenceProvider(max_concurrent_inferences=1)

mock_model = MagicMock()
mock_tokenizer = MagicMock()

mock_output = MagicMock()
mock_output.sequences = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
mock_model = MagicMock()
mock_model.generate.return_value = MagicMock(sequences=[[1,2,3,4,5,6,7,8,9,10]])

mock_inputs = MagicMock()
mock_inputs.input_ids = MagicMock()
mock_inputs.input_ids.shape = (1, 3)
mock_inputs.to = MagicMock(return_value=MagicMock(input_ids=MagicMock(shape=(1, 3))))

mock_tokenizer = MagicMock()
mock_tokenizer.return_value = mock_inputs
mock_tokenizer.decode.return_value = "test response"
mock_tokenizer.eos_token_id = 2

provider = InferenceProvider(max_concurrent_inferences=1)
provider.ensure_loaded = MagicMock(return_value=(None, None))  # Will be set per test

# Test the mock
print("Before mock:", provider.ensure_loaded)
provider.ensure_loaded = MagicMock(return_value=(None, None))
print("After mock:", provider.ensure_loaded)
result = provider.ensure_loaded('fp16')
print("Result:", result)

# Now test with actual mocks
provider.ensure_loaded = MagicMock(return_value=(None, None))
result = provider.ensure_loaded('fp16')
print("Result:", result)