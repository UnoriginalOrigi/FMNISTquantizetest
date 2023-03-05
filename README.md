# FMNISTquantizetest

Fashion-MNIST tested on a simple LR model with quantization (resized to 8x8 from 28x28 and pixel values reduced from [0; 255] (8-bit) to [0;7] (3-bit)) using pyTorch. Reporting accuracy after 50 epochs with 0.0001 learning rate (10 batch size). Tested on Windows with i5-8600K

No quantization:

Total training time: 634.487 s

Best accuracy: 84.91 %

![Picture1](https://user-images.githubusercontent.com/105780035/222964994-bbd02358-71c8-43fb-af37-85206a634f22.png)

![Picture2](https://user-images.githubusercontent.com/105780035/222965003-0fc589d5-9e89-45d4-8bd2-968893c2a11c.png)


Quantized:

Total training time: 688.315 s

Best accuracy: 67.98 %

![Picture3](https://user-images.githubusercontent.com/105780035/222965030-43a8cf88-923d-4f71-a720-4ff43283462e.png)

![Picture4](https://user-images.githubusercontent.com/105780035/222965032-6db8f990-8993-4e0b-9438-cf413d86bf33.png)

Extra info:
ToTensor automatically normalizes the data to be in range [0; 1] by assuming pixel value to be around 255 for FMNIST. Trying to avoid the normalization throws errors as soon as training starts
