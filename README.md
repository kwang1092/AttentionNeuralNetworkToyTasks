# AttentionNeuralNetworkToyTasks
Implemented Attention into our RNNs to complete simple "toy" tasks

Goal: The goal of this project was to view the effectiveness of implementing Attention into our seq2seq "toy" tasks and see which form of attention performed best(additive or multiplicative attention)

These toy tasks consisted of copying, reversing, and sorting integer arrays using neural networks. Although this could have been easily done with simple algorithms, I wanted to see if neural networks could be used.

For additive attention, I decided to use to the ReLU activation
function over tanh, because it seems to be the more predominantly used activation
function for deep learning applications. For both attention implementations, I took in hidden states from
the decoder from the previous time step and also used a for-loop to loop through each individual
encoder state. Overall, both implementations seemed to be much more effective for copying and reversing, with
additive attention being the more accurate version. 

Analysis: 

In terms of evaluation time, running the model without attention was significantly faster than
with attention. In addition, additive attention took significantly longer than multiplicative attention. This
was expected because of the used inner for loops to implement attention, so this would take longer than no
attention; in a certain step, additive attention involved two matrix multiplications and an activation
function while multiplicative attention only involved one matrix multiplication, so it is reasonable that
additive attention would take longer than multiplicative attention. For copying and reversing, my
accuracy increased significantly by using attention. I noticed that additive attention was better than
multiplicative attention for these tasks. For sorting, attention decreased accuracy slightly; this could be
because attention weighs on input order, but, with sorting, the input order has no effect on the final
output. In addition, as shown in the example above, I noticed that the model without attention often had
inaccurate predictions, the additive attention model was often completely correct in its predictions, and
the multiplicative attention model was often mostly correct with the exception of the first output of the
sequence.

Evaluation Time (in seconds):

| | No attention| Attention (additive) | Attention (multiplicative) |
|-|-------------|----------------------|----------------------------|
| Copying |  11.17 s | 35.84 s| 27.04 s|
|Reversing |10.65 s |44.84 s |24.12 s|
|Sorting |9.15 s |34.85 s |27.99 s|

Accuracy:

| |No attention |Attention (additive) |Attention (multiplicative)|
|-|-------------|---------------------|--------------------------|
|Copying| 59.54%| 99.73% |87.52%|
|Reversing |80.3% |97.17% |90.29%|
|Sorting |98.36% |93.10% |91.23%|
