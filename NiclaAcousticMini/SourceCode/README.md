
# SourceCode acoustic data Tiny model classifier

This folder contains the PyTorch source code I use to train and test an
acoustic classifier for my **Bioacoustics_of_IoT / NiclaAcousticMini** project.

The code is designed for an ESC-50 style dataset that has been prepared in
an **Edge Impulse–like** folder structure.

---

## Dataset layout

On my machine the data is stored here:

```text
C:\Users\bkarimov\OneDrive - University of Tasmania\Desktop\ESC-50-master\
  renamed\
    training\
      class_1\*.wav
      class_2\*.wav
      ...
    testing\
      class_1\*.wav
      class_2\*.wav
      ...
