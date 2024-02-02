import datetime
import subprocess
 
training_setups = [
    r'torchrun --nproc_per_node 2 hugging-trainer.py --name="MaskFormer" --model="facebook/maskformer-swin-tiny-ade" --epochs=1 --batch-size=46 --patch-size=256 --lr=1e-3',
    #r'torchrun --nproc_per_node 2 hugging-trainer.py --name="MaskFormer" --model="facebook/maskformer-swin-tiny-ade" --epochs=500 --batch-size=4 --patch-size=640 --lr=1e-3',
    #r'torchrun --nproc_per_node 2 hugging-trainer.py --name="MaskFormer" --model="facebook/maskformer-swin-tiny-ade" hugging-trainer.py --epochs=5 --batch-size=4 --patch-size=800 --lr=1e-3',

    #r'./.venv/bin/python main.py --epochs=10 --batch-size=42 --patch-size=256 --lr=1e-2',
    #r'./.venv/bin/python main.py --epochs=500 --batch-size=8 --patch-size=640 --lr=1e-4 --save-checkpoints',
    #r'./.venv/bin/python main.py --epochs=500 --batch-size=3 --patch-size=800 --lr=1e-4 --save-checkpoints',
]
 
for training in training_setups:
    start_time = datetime.datetime.now()
    subprocess.call(training, shell=True)
    end_time = datetime.datetime.now()
    print(f"Training run took: {end_time - start_time}.")