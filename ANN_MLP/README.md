# Simple MLP for adding two numbers

Files:
- `model.py` — MLP class and save/load helpers
- `trainer.py` — train the model and save to `model.pth`
- `inference.py` — load model and test on hardcoded values

Quick run:

1. Install requirements:

```powershell
pip install -r requirements.txt
```

2. Train (short example):

```powershell
python trainer.py --epochs 100 --batch_size 512 --lr 1e-3 --save model.pth
```

3. Test:

```powershell
python inference.py
```
