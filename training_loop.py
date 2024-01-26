import datetime
import subprocess

training_setups = [
    r'./.venv/bin/python main.py --epochs=500 --batch-size=160 --patch-size=256 --weight-decay=1e-3 --lr=1e-4 --classes=2 --use-amp --use-ddp --save-checkpoints',
    r'./.venv/bin/python main.py --epochs=500 --batch-size=18 --patch-size=640 --weight-decay=1e-3 --lr=1e-4 --classes=2 --use-amp --use-ddp --save-checkpoints',
    r'./.venv/bin/python main.py --epochs=500 --batch-size=10 --patch-size=800 --weight-decay=1e-3 --lr=1e-4 --classes=2 --use-amp --use-ddp --save-checkpoints',
]

for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training, shell=True)
    end_time = datetime.datetime.now()

    print(f"Training run took: {end_time - start_time}.")