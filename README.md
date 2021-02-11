# LeopardGecko
This package contains routines to process and analyse prediction data obtained from predicted segmentation of microscopy data in package.
See  [DiamondLightSource / placental-segmentation-2dunet](https://github.com/DiamondLightSource/placental-segmentation-2dunet/blob/main/blood_vessels/placenta_blood_vessel_2d_unet_prediction.ipynb).

The idea is that combined data from the notebook lacenta_blood_vessel_2d_unet_prediction.ipynb is analysed in terms of consisteny of the prediction from the segmentation AI used.

## AvgPooling3DConsistencyData.ipynb
An average pooling operation using PyTorch CUDA routines are used. The data obtained gives a score about how consistent the data is in the each of smaller volumes.

## AnalyseAvgPoolResults.ipynb
Data from the average pooling is analysed and consistency score based visualisations are available

![LeopardGecko meme](./LeopardGeckoGitMeme.png)