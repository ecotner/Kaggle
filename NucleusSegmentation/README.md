# Nucleus Segmentation (Kaggle Data Science Bowl 2018)

The subject of this year's data science bowl is to use computer vision techniques to create segmentation maps of nuclei in microscope images. This is actually _instance_ segmentation, not semantic segmentation, so each nucleus must be individually identifiable as its own distinct object.

### To do:
* Overhaul preprocessing pipeline to include interpolation layer
* Change U-Net loss to accommodate mixed xentropy/MSE loss
* Change prediction/inference script to be able to decipher new output

### Done:
* Write prediction/inference script
* Make training routine
* Make data preprocessing pipeline
* Make U-Net
