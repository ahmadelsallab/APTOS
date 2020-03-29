# [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
___Detect diabetic retinopathy to stop blindness before it's too late___

## What is Diabetic Retinopathy (DR)?

<p align="center">
  <img src="imgs/DR.png" />
</p>

_Source: [Medical Diagnosis with a Convolutional Neural Network, TowardsDataScience](https://medium.com/m/global-identity?redirectUrl=https%3A%2F%2Ftowardsdatascience.com%2Fmedical-diagnosis-with-a-convolutional-neural-network-ab0b6b455a20)_


According to [this article](https://www.diabetes.co.uk/diabetes-complications/diabetic-retinopathy.html):

- Diabetic retinopathy is the most common form of diabetic eye disease. Diabetic retinopathy usually only affects people who have had diabetes (diagnosed or undiagnosed) for a significant number of years.
- Retinopathy can affect all diabetics and becomes particularly dangerous, increasing the risk of blindness, if it is left untreated.
-The risk of developing diabetic retinopathy is known to increase with age as well with less well controlled blood sugar and blood pressure level.
- According to the NHS, 1,280 new cases of blindness caused by diabetic retinopathy are reported each year in England alone, while a further 4,200 people in the country are thought to be at risk of retinopathy-related vision loss.
All people with diabetes should have a dilated eye examination at least once every year to check for diabetic retinopathy.

## Why Computer Vision (CV) for DR diagnosis?

According to the [APTOS Kaggle Comptetion home](https://www.kaggle.com/c/aptos2019-blindness-detection) page:

_Millions of people suffer from diabetic retinopathy, the leading cause of blindness among working aged adults. Aravind Eye Hospital in India hopes to detect and prevent this disease among people living in rural areas where medical screening is difficult to conduct._

___The need to AI:___

_Currently, Aravind technicians travel to these rural areas to capture images and then rely on highly trained doctors to review the images and provide diagnosis. Their goal is to scale their efforts through technology; to gain the ability to automatically screen images for disease and provide information on how severe the condition may be._

![APTOS_AI](imgs/APTOS_AI.png)

## How doctors diagnose DR?

According to [www.eyeops.com](https://www.eyeops.com/contents/our-services/eye-diseases/diabetic-retinopathy), doctors look for at least 5 patterns as in the image below:

![DR_cotton_wool](https://sa1s3optim.patientpop.com/assets/images/provider/photos/1947516.jpeg)

_Diabetic retinopathy can result in many serious issues affecting the blood vessels that nourish the retina., Source [https://www.eyeops.com/](https://www.eyeops.com/contents/our-services/eye-diseases/diabetic-retinopathy)_

_Diabetic retinopathy occurs when the damaged blood vessels leak blood and other fluids into your retina, causing swelling and blurry vision. The blood vessels can become blocked, scar tissue can develop, and retinal detachment can eventually occur._

## AI that explains itself

In this work, I was particularly interested in using AI and CV (mainly ConvNets), and visualize the learnt patterns by the ConvNet feature maps, to see if similar patterns as above (Cotton wool spots, Hemorrhages, hard Exudates, Aneurysm and Abnormal growth of blood vessels), are also detected by ConvNets?

![APTOS_AI_Explanation](imgs/APTOS_AI_Explanation_2.png)

# Data

Classes are:
0 - No DR

1 - Mild

2 - Moderate

3 - Severe

4 - Proliferative DR


__Class imabalance__

Before starting, very basic EDA shows class imabalnce issue:
![Class_imbalance](imgs/Class_imbalance.png)

This will be treated later.


# Evaluation metric
The metric used is Quadratic Weighted Kappa ([QWKP](https://www.kaggle.com/c/aptos2019-blindness-detection/overview/evaluation)). It is well explained in this [kernel](https://www.kaggle.com/xhlulu/aptos-2019-densenet-keras-starter).

The evaluation metric is of particular interest. The essence of QWKP is to favor prediction mistakes that are close to the correct answer than the ones far from it. In other words, if the correct class is "Mild", while the prediction is "Moderate", this is better than if the prediction is "Severe". This is intuitive, specially for "Ordinal" target variables, where classes represent a mark on an ordered discrete scale, representing severity as in our case. This is a common case in many medical diagnosis problems, like in Radiology or Lab reports for example.

If you navigate in the Kaggle kernels of APTOS competetion, you will see three main approaches to specify the loss function and network output. This relates to problem formulation as one of the following:

| Problem | Loss | Network output | Comment|
|---------|:-----|:---------------|:-------|
| Multi-class | Cross Entropy | Softmax/Class probabilities | Normal choice. But not good for QWKP, since CE favors only the correct class|
| Regression | RMSE | Linear/Relu | If RMSE is small enough, this is good since the error/confusion will at max to the neighbor class|
| Multi-label | Binary Cross Entropy | Sigmoid | By formatting the ground truth labels such that the label is all ones until the correct prediction then all 0's. This encourages the model to output the correct class or at least the neighboring ones|

The [wikipedia page](https://en.wikipedia.org/wiki/Cohen%27s_kappa) offer a very concise explanation:
_"The weighted kappa allows disagreements to be weighted differently and is especially useful when codes are ordered. Three matrices are involved, the matrix of observed scores, the matrix of expected scores based on chance agreement, and the weight matrix. Weight matrix cells located on the diagonal (upper-left to bottom-right) represent agreement and thus contain zeros. Off-diagonal cells contain weights indicating the seriousness of that disagreement."_



# Model
## ConvNet model

Some ideas in the code are insipred by this Kaggle [kernel](https://www.kaggle.com/mathormad/aptos-resnet50-baseline).


## Small custom model

As a start, trying a simple small Conv2D model seems to perform fairly good:

```
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(sz, sz, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(n_classes, activation='softmax'))
```

## ResNet50 model

## Optimization
### ReduceLROnPlateau
### Early stopping
### CyclicalLR
## QWKP metric
