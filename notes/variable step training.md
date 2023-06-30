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

