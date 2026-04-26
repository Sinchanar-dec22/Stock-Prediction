import sys
print("Python path:", sys.path[:3])
try:
    import numpy as np
    print("✓ numpy")
except Exception as e:
    print(f"✗ numpy: {e}")
    
try:
    import pandas as pd
    print("✓ pandas")
except Exception as e:
    print(f"✗ pandas: {e}")
    
try:
    from sklearn.linear_model import LinearRegression
    print("✓ sklearn.linear_model")
except Exception as e:
    print(f"✗ sklearn.linear_model: {e}")
    
try:
    from sklearn.preprocessing import MinMaxScaler
    print("✓ sklearn.preprocessing")
except Exception as e:
    print(f"✗ sklearn.preprocessing: {e}")
    
try:
    from sklearn.metrics import mean_squared_error
    print("✓ sklearn.metrics")
except Exception as e:
    print(f"✗ sklearn.metrics: {e}")
    
try:
    import math
    print("✓ math")
except Exception as e:
    print(f"✗ math: {e}")

print("\nNow trying to load model.py...")
try:
    from model import train_and_predict
    print("✓ train_and_predict imported!")
except Exception as e:
    print(f"✗ Failed to import train_and_predict: {e}")
    import traceback
    traceback.print_exc()
