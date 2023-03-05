# FMNISTquantizetest

Fashion-MNIST tested on a simple LR model using pyTorch. Reporting accuracy after 50 epochs with 0.0001 learning rate (10 batch size). Tested on Windows with i5-8600K

No quantization:

Total training time: 634.4874987602234 s

Best accuracy: 84.91000366210938 %

![Picture1](https://user-images.githubusercontent.com/105780035/222964994-bbd02358-71c8-43fb-af37-85206a634f22.png)

![Picture2](https://user-images.githubusercontent.com/105780035/222965003-0fc589d5-9e89-45d4-8bd2-968893c2a11c.png)


Quantized:

Total training time: 688.3150005340576

Best accuracy: 67.9800033569336

![Picture3](https://user-images.githubusercontent.com/105780035/222965030-43a8cf88-923d-4f71-a720-4ff43283462e.png)

![Picture4](https://user-images.githubusercontent.com/105780035/222965032-6db8f990-8993-4e0b-9438-cf413d86bf33.png)
