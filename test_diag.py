import sys; sys.path.insert(0, '.');
from backend.main import load_model_diagnostics;
import traceback
print('testing')
try:
    load_model_diagnostics()
except Exception as e:
    traceback.print_exc()
