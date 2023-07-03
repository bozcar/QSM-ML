# Variable Step Training

## 27/06/23

| Parameter               | Value |
|-------------------------|-------|
| Iterations              | 200   |
| Initial Learning Rate   | 0.5   |
| LR reduction on Plateau | 0.75  |
| Plateau Patience        | 25    |
| Early Stopping Patience | 500   |

Weights all ended up around 0.4, slightly smaller on first layers, larger later on. No noticable early-stopping. Visually inspected the image produced, very bad, worse than untrained. Maybe overfitted to the training data???

## 03/07/23

First test run using data from SL-QSM. Mostly testing whether data cleaning was successful

| Parameter               | Value |
|-------------------------|-------|
| Iterations              | 200   |
| Initial Learning Rate   | 0.5   |
| Max Epochs              | 10    |

With current settings/optimisation/hardware, epochs take 10 - 11 mins.

Epoch 4 evaluated loss to nan. Perhaps learning rate too high??? Not sure.

Rerun test with lower learning rate of 0.01 (rechecked Adam literature, lower learning rates usually preferred).

| Parameter               | Value |
|-------------------------|-------|
| Iterations              | 200   |
| Initial Learning Rate   | 0.01  |
| Max Epochs              | 5     |
