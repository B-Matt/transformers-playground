import datetime
import subprocess
 
training_setups = [
    #r'./.venv/bin/python main.py --epochs=500 --batch-size=42 --patch-size=256 --weight-decay=1e-3 --lr=1e-4 --save-checkpoints',
    #r'./.venv/bin/python main.py --epochs=500 --batch-size=8 --patch-size=640 --weight-decay=1e-3 --lr=1e-4 --save-checkpoints',
    r'./.venv/bin/python main.py --epochs=500 --batch-size=3 --patch-size=800 --weight-decay=1e-3 --lr=1e-4 --save-checkpoints',
]
 
for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training, shell=True)
    end_time = datetime.datetime.now()
    print(f"Training run took: {end_time - start_time}.")