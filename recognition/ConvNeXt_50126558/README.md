Comparison between 2 models (ConvNeXt and Basic MLP)

Chosen Baseline Model - Basic MLP
All the other variables are controlled to have a fair comparison. 
Controlled variables
1) Seed = 0
2) Epoch = 30
3) Batch size = 32
4) Optimizer = Adam
5) Parameters = 27.82M 
6) Learning Rate = 1e-4
7) Loss Function = Cross Entropy Loss
8) Train/val/test split (20% of train patients for validation)
9) Preprocessing and Augmentations (imported from dataset.py to ensure that it is same)
10) Evaluation matrix (loss, accuracy, AUROC, classfication report, confusion matrix, peak VRAM)
11) Model selection = the checkpoint with the highest validation accuracy was selected, and that checkpoint was evaluated on the test set.

Trial and Errors
Comparing between 2 models
1) Chose Resnet-50 as baseline as it can be modified from demo 2 Resnet-18 because it has the same structure
2) Result: Resnet has a higher test accuracy compare to ConvNeXt
3) Confusion: Unclear if all parameters need to be same including normalization technique, activation function and others
4) Realization: Resnet-50 shouldn't be used as a baseline comparison as the question only ask for a standard convolutional network, MLP, or basic linear classifier
5) A basic MLP is then chosen as the comparison
6) The parameters are set to same for comparison as shown above. 

Results
                                ConvNeXt            Basic MLP
Best Validation Accuracy        0.8609              0.6898
test accuracy                   0.7366              0.6207
test AUROC                      0.8176              0.6384
peak VRAM                       2.68 GB             0.58 GB

The results show that ConvNeXt outperformed the Basic MLP in validation accuracy, test accuracy, and test AUROC. Test accuracy increased from 62.07% to 73.66%, representing an improvement of 11.59 percentage points. Test AUROC increased from 0.6384 to 0.8176, indicating that ConvNeXt had a stronger ability to distinguish between the NC and AD classes across different classification thresholds.

ConvNeXt also required more peak GPU memory, using 2.68 GB compared with 0.58 GB for the Basic MLP. This indicates a higher memory requirement under the tested configurations, despite both models having approximately the same number of parameters.

Overall, ConvNeXt achieved stronger classification performance than the Basic MLP on the reported accuracy and AUROC metrics, at the cost of higher GPU memory usage. These results suggest that the ConvNeXt is better suited to this image-classification task under the experimental conditions used.

Terminologies
1) Train set - the data used to train the model by calculating gradients and updating its weights.
2) Validation set - the data used to evaluate the model to improve during the training
3) Test set - the data used to test how well the model perform after training
4) Seed - a number used to control random number generation so that experiments can be reproduced more consistently.
5) Epoch - one complete pass through all training examples.
6) Batch size - number of training samples processed together in one training step. Larger - faster processing which requires more memory; Smaller - use lesser memory, more iterations and optimizer updates per epoch
7) Optimizer - the algorithm that update the model's weights (e.g. SGD, Adam)
8) Learning Rate - control the size fo optimizer's parameter updates. 