import sys
print(f'Python: {sys.version}')

deps = {}
try:
    import numpy as np
    deps['numpy'] = np.__version__
except ImportError as e:
    deps['numpy'] = f'MISSING: {e}'

try:
    import torch
    deps['torch'] = torch.__version__
    deps['cuda_available'] = str(torch.cuda.is_available())
    if torch.cuda.is_available():
        deps['cuda_version'] = torch.version.cuda
        deps['gpu'] = torch.cuda.get_device_name(0)
except ImportError as e:
    deps['torch'] = f'MISSING: {e}'

try:
    from torch.utils.data import DataLoader, Dataset
    deps['torch.utils.data'] = 'OK'
except ImportError as e:
    deps['torch.utils.data'] = f'MISSING: {e}'

try:
    import torch.nn as nn
    deps['torch.nn'] = 'OK'
except ImportError as e:
    deps['torch.nn'] = f'MISSING: {e}'

deps['json/pathlib'] = 'OK (stdlib)'

for k, v in deps.items():
    print(f'  {k}: {v}')
