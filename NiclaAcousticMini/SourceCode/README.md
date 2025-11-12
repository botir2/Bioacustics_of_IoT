# SourceCode acoustic data tiny model classifier

This folder contains the PyTorch source code I use to train and test a
tiny acoustic classifier for my **Bioacoustics_of_IoT / NiclaAcousticMini** project.

The code is designed for a dataset that has been prepared in a suitable folder structure.

---

## Files in this folder

- `SourceCode.py` – main training and evaluation script  
- `runTest.py` – script to evaluate a saved model on the testing split  
- `requirements.txt` – Python dependencies needed to run the code  

python SourceCode.py --mode train --epochs 30 --batch_size 8
or
python SourceCode.py --mode eval --model_path esc50_cnn_best.pt


python runTest.py --data_root C:"<Your_dir>" --model_path esc50_cnn_best.pt

---

## Dataset layout

On my machine the data is stored here:

```text
C:"<Your dir>"
  renamed\
    training\
      class_1\*.wav
      class_2\*.wav
      ...
    testing\
      class_1\*.wav
      class_2\*.wav
      ...

