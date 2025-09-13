import subprocess
import os

def test_preprocess_and_train():
    res = subprocess.run(["python", "src/preprocess.py"])
    assert res.returncode == 0
    res2 = subprocess.run(["python", "src/train.py"])
    assert res2.returncode == 0
    assert os.path.exists("models/model.joblib")
