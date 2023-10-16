Variable Step Training
======================

27/06/23
--------

| Parameter               | Value |
|-------------------------|-------|
| Iterations              | 200   |
| Initial Learning Rate   | 0.5   |
| LR reduction on Plateau | 0.75  |
| Plateau Patience        | 25    |
| Early Stopping Patience | 500   |

Weights all ended up around 0.4, slightly smaller on first layers, larger later on. No noticable early-stopping. Visually inspected the image produced, very bad, worse than untrained. Maybe overfitted to the training data???

03/07/23
--------

First test run using data from SL-QSM. Mostly testing whether data cleaning was successful

| Parameter               | Value |
|-------------------------|-------|
| Iterations              | 200   |
| Initial Learning Rate   | 0.5   |
| Max Epochs              | 10    |

With current settings/optimisation/hardware, epochs take 10 - 11 mins.

Epoch 4 evaluated loss to nan. Perhaps learning rate too high??? Not sure.

-----------------------------------------------------------------------------

Rerun test with lower learning rate of 0.01 (rechecked Adam literature, lower learning rates usually preferred).

| Parameter               | Value |
|-------------------------|-------|
| Iterations              | 200   |
| Initial Learning Rate   | 0.01  |
| Max Epochs              | 5     |

Run did not produce nan values in first 5 epochs, seems the lower LR is better with the larger dataset.

-----------------------------------------------------------------------------

Longer run with dataset shuffling and tensorboard logging added.

| Parameter               | Value |
|-------------------------|-------|
| Iterations              | 200   |
| Initial Learning Rate   | 0.01  |
| LR reduction on Plateau | 0.75  |
| Plateau Patience        | 5     |
| Early Stopping Patience | 10    |
| Max Epochs              | 20    |

Shuffle buffer does take some time to fill between epochs, haven't timed it, feels like less than a minute per epoch. Maybe remove the shuffle on the validation data?

07/07/23
--------

Doing a long run to get outputs for the presentataion.

Have topped up the training data to replace corrupt files. One new file deleted during data cleaning.

Removed shuffle on validation data. 

| Parameter               | Value |
|-------------------------|-------|
| Iterations              | 200   |
| Initial Learning Rate   | 0.001 |
| LR reduction on Plateau | 0.5   |
| Plateau Patience        | 3     |
| Early Stopping Patience | 9     |
| Max Epochs              | 100   |

12/07/23 :rage:
---------------

Aaaarghhhh!!!! It's all broken, I have to recheck everything!!!!! :rage::rage::rage:

Acutally, the only think I know is broken is `VariableStepNDI`, but that's annoying enough on it's own.

### What I think the model should be doing

We are looking for the minimum of:

$$f(\chi) = ||W(\exp(iD\chi)-\exp(i\phi))||^2_2$$

Iterative scheme:

$$\chi_0 = 0$$

$$\chi_{k+1} = \chi_k - \tau_k D^T W^T W \sin(D\chi_k - \phi)$$

Before training the model $\tau_k = 2, \forall k \isin [0, N-1)$ where N is the number of iterations. $\tau$ is a trainable model parameter.

### I have now looked through the whole model

I have now looked through the whole model and I can't see anything which is obviously wrong. All the componenets seem to do what I think they should do. The only thing which might be causing the problems is that I am assuming that W will always be the identity.

I don't think this is an issue however, as I was previously always passing ones in as my W matrix. This would have been executing the identity operation.

I guess I'll have to do tests of each model component to see if there's any weird behaviour.

To watch the saga unfold, take a look at: `./checking_model_components.ipynb`. If you just care about the result, keep reading.

### I think I've found the problem

The long and short of my tests seems to show that the issue I'm having is (once again...) that my dipole kernel is facing the wrong direction. :unamused:

Here's how the dipole kernel is supposed to work:

The dipole kernel represents the magnetic field of a point susceptibility source, to which a magnetic field has been applied. In MRI, the magnetic field we care about is the one that the scanner applies to the subject - the $B_0$ field.

Cannonically, the $B_0$ field is oriented in the z direction.

In MRI, we often switch between the image-sapce and k-space views of objects and so I use the convention of calling the image-space dipole kernel $D(\vec{r})$ and the k-space dipole kernel $d(\vec{k})$. As it happens, $d(\vec{k})$ is the much more useful form.

Assuming $\vec{B_0} = B_z\hat{\mathbf{z}}$, then $d(\vec{k})$ is given by:

$$d(\vec{k}) = \frac{1}{3}-\frac{k_z^2}{\vec{k}^2}$$

(For whatever reason people always seem to feel the need to point out that $\vec{k}^2 := \vec{k}\cdot\vec{k} = k_x^2 + k_y^2 + k_z^2$, so I guess I will here too.)

A useful note is that, along the z-axis:

$$\vec{k} = k_z\hat{\mathbf{e_z}} \implies d(\vec{k}) = -\frac{2}{3}$$

In this project, the z-axis can always be assumed to be in the axial direction and so in a sagital slice, the z-axis should be oriented vertically.

### And here's the evidence

When I checked the kernel being used in the `ConvDipole` layer, this is what I found:

![](/notes/images/bad%20dipole.png)

And here's what it should have looked like:

![](/notes/images/good%20dipole.png)

We can clearly see that the dark areas with values around -2/3 which should oriented vertically are instead horizontal; indicating that the dipole kernel is oriented in the coronal direction, not the axial one.

Taking a coronal slice through the iamge confirms the problem:

![](/notes/images/bad_coronal_slice.png)

This image ought to look the same as the first. Instead, it is clear that what ought to be the z-axis is oriented normal to the image plane.

Now I've just got to figure out how to fix it.

### I fixed it!

Looking through the `generate_dipole` method of `ConvDipole`, I discovered this line:

```
z_squared = vy**2
```

This is clearly wrong and has now been corrected to:

```
z_squared = vz**2
```

And so the saga concludes.

------

Now that it's fixed, I'll set off a run to see how that goes. Just going to keep the settings I had on the last run.

| Parameter               | Value |
|-------------------------|-------|
| Iterations              | 200   |
| Initial Learning Rate   | 0.001 |
| LR reduction on Plateau | 0.5   |
| Plateau Patience        | 3     |
| Early Stopping Patience | 9     |
| Max Epochs              | 100   |

13/07/23
--------

### Overnight run results

The run I set off last night has concluded. Seems to have run fine. From the console, the final `val_loss` is 3.5e-03 and the run stopped on epoch 15. 

The learned step-size is ~0.264 for all steps. (IDEA: train a model with 26 iters to see if the learned step is near 2)

Visually, the reconstruction looks good. Susceptibility values are in a reasonable range. Looks like an intermediary between 10 and 50 iterations of untrained. (IDEA: generate image of untrained with 26 iters and visually compare)

Should do some numerical comparisons to GT.

Created image with untrained, 26 iters. Visually very similar to learned image.

-------
### More runs to get some results with the fixed model

After talking with Patrick about what results to get for the project talk we came up with:

1. Varying initial learning rate
2. Trying initialising with random weights
3. Varing number of iterations
4. Investigating different methods of LR reduction

First, going to try LR changes. Did one run in the day:

| Parameter                 | Value |
|---------------------------|-------|
| Iterations                | 200   |
| Initial Learning Rate     | 0.01  |
| LR reduction on Plateau   | 0.5   |
| Plateau Patience          | 3     |
| Early Stopping Patience   | 9     |
| Max Epochs                | 100   |

Training converged after 15 epochs, very similar losses as the previous run.

----
### Some thoughts on possible optimisation

I'm also thinking about possible performance optimisations. I think the two things most likely to cause a bottleneck are I/O and GPU memroy bandwidth.

In terms of I/O, I think the shuffle buffer works the same as a regular I/O buffer, so I don't think I'm going to be I/O bottlenecked yet. Filling the shuffle buffer however does currently take an appreciable ammount of time between epochs, ~10 secs I think. Apparently I can speed this up with interleaving, which I could look into. 

Could also try increasing batch size to speed up training. This would be more demanding both of the I/O and the memroy bandwidth. In terms of GPU memory I think I have a reasonable ammount more to play with. 

Current plan for training optimisations:

1. Start increasing batch size and see how it affects GPU memory usage
    - might be able to max that out without any other changes
2. See if I can work out how interleaving works
    - if it's easy, I don't see any reason not to implement it
    - however, don't put too much effort in unless obviously I/O bottlenecked

----

And an overnight run with a really low LR

| Parameter                 | Value |
|---------------------------|-------|
| Iterations                | 200   |
| Initial Learning Rate     | 0.0001|
| LR reduction on Plateau   | 0.5   |
| Plateau Patience          | 3     |
| Early Stopping Patience   | 9     |
| Max Epochs                | 100   |

14/07/23
--------

### Results of overnight run

Took **ages** to converge (38 epochs)

16/07/23
--------

| Parameter                 | Value |
|---------------------------|-------|
| Iterations                | 200   |
| Initial Learning Rate     | 0.1   |
| LR reduction on Plateau   | 0.5   |
| Plateau Patience          | 3     |
| Early Stopping Patience   | 9     |
| Max Epochs                | 100   |

Reached plateau after 1 interation

----

| Parameter                 | Value |
|---------------------------|-------|
| Iterations                | 200   |
| Initial Learning Rate     | 1     |
| LR reduction on Plateau   | 0.5   |
| Plateau Patience          | 3     |
| Early Stopping Patience   | 9     |
| Max Epochs                | 100   |

loss -> nan on epoch 4

----

Try random initialisation:

| Parameter                 | Value     |
|---------------------------|-----------|
| Iterations                | 200       |
| Weight Initialisation     | random    |
| Initial Learning Rate     | 0.0001    |
| LR reduction on Plateau   | 0.5       |
| Plateau Patience          | 3         |
| Early Stopping Patience   | 5         |
| Max Epochs                | 100       |

set initilal LR too low (oops)

17/07/23
--------

More random initialisation runs (this time with correct LR initialisation!!!)

| Parameter                 | Value     |
|---------------------------|-----------|
| Iterations                | 200       |
| Weight Initialisation     | random    |
| Initial Learning Rate     | 0.001     |
| LR reduction on Plateau   | 0.5       |
| Plateau Patience          | 3         |
| Early Stopping Patience   | 5         |
| Max Epochs                | 100       |

06/08/23
--------

| Parameter                 | Value     |
|---------------------------|-----------|
| Iterations                | 200       |
| Weight Initialisation     | random    |
| Initial Learning Rate     | 0.003     |
| LR reduction on Plateau   | 0.5       |
| Plateau Patience          | 3         |
| Early Stopping Patience   | 5         |
| Max Epochs                | 100       |
