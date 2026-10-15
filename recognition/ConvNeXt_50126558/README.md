ConvNeXt

Baseline Model Comparison - Basic MLP
All the other variables are controlled to have a far comparison. 
Controlled variables
1) Seed = 0
2) Epoch = 30
3) Batch size = 32
4) Optimizer = Adam
5) Parameters = 27.82M 
6) Learning Rate = 1e-4
7) Loss Function = Cross Entropy Loss
8) Preprocessing and Augmentations (imported from dataset)
9) Train/val/test split (20% of train patients for validation)
10) Model selection = test point with best validation accuracy
11) Evaluation matrix (test loss, accuracy, AUROC, classfication report, confusion matrix, peak VRAM)