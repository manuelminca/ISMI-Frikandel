# ISMI-Frikandel

## Abstract
In this work we describe our participation in the medical decathlon challenge for the specific task of pancreas segmentation.
The objective is to segment pancreas and cancer from CT-Scans. In order to do this, Neural Networks have been used to create these predictions. Specifically, the 3D-Unet model has been used due to its state-of-the-art results. In total, the model has been trained for approximately 60 hours for a total of 22 epochs. An auxiliary dataset containing patches extracted from the original scans has been generated and used to train the model.
The segmentation was somewhat successful. We obtain dice scores of 0.99, 0.63 and 0.28 for background, pancreas and cancer respectively on validation patches and similar scores with a less trained model on complete test volumes. Unfortunately, we were not able to submit predictions for the test volumes with our final model due to technical problems at grand challenge.

## Additional Comments

This work has been developed by N. van den Hork, M. Mínguez Carretero, M. Moons and M. Schilpzand for Intelligent Systems for Medical Imaging course at Radboud University. 
