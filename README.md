> [!NOTE]  
> November 2024: New Pre-trained Models are available, check the [Pre-trained Models](#pre-trained-models) section.

# Foggy-CycleGAN

<p align="center">
 <img src="images/banner-cropped-rnd.png">
</p>

This project is the implementation for my Computer Science MSc thesis in the University of Debrecen.

Dissertation: 
<a href="./dissertation/Simulating%20Weather%20Conditions%20on%20Digital%20Images%20-%20Final.pdf" target="_blank">[PDF] Simulating Weather Conditions on Digital Images</a> (Debrecen, 2020).

# Table of Content
- [Description](#description)
- [Code](#code)
- [Notebook](#notebook)
- [Results](#results)
- [Pre-trained Models](#pre-trained-models)

## Description
**Foggy-CycleGAN** is a
<a href="https://junyanz.github.io/CycleGAN/" target="_blank">CycleGAN</a> model trained to synthesize fog on clear images. More details in the dissertation above.

## Code
The full source code is available under GPL-3.0 License in my Github repository <a href="https://github.com/ghaiszaher/Foggy-CycleGAN" target="_blank">ghaiszaher/Foggy-CycleGAN</a>

## Notebook <a href="https://colab.research.google.com/github/ghaiszaher/Foggy-CycleGAN/blob/master/Foggy_CycleGAN.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
A Jupyter Notebook file <a href="https://github.com/ghaiszaher/Foggy-CycleGAN/blob/master/Foggy_CycleGAN.ipynb" target="_blank">Foggy_CycleGAN.ipynb</a> is available in the repository.

## Results
(as of June 2020)
<p align="center">
 <img src="images/results/2020-06/result-animated-01.gif">
</p>

<p align="center">
 <img src="images/results/2020-06/result-sample-0.2.jpg">
</p>

<p align="center">
 <img src="images/results/2020-06/result-sample-0.3.jpg">
</p>

<p align="center">
 <img src="images/results/2020-06/result-sample-0.25.jpg">
</p>

<div align="center">
&copy; Ghais Zaher 2020
</div>

## Pre-trained Models
As previous pre-trained models are no longer compatible with newer Keras/Tensorflow versions, I have retrained the model and made the new weights available to download.

Each of the following models was trained in Google Colab using the same dataset, the parameters for building the models and number of trained epochs are a bit different:
<div align="center">

| Model                                                                                                    | Trained Epochs | Config                                                                              |
|----------------------------------------------------------------------------------------------------------|----------------|-------------------------------------------------------------------------------------|
| [2020-06 (legacy)](https://drive.google.com/drive/folders/1QKsiaGkMFvtGcp072IG57MfY1o_D-L3k?usp=sharing) | 292            | `use_transmission_map=False`<br>`use_gauss_filter=False`<br>`use_resize_conv=False` |
| 2024-11-17-rev1-000                                                                                      | 178            | `use_transmission_map=False`<br>`use_gauss_filter=False`<br>`use_resize_conv=False` |
| 2024-11-17-rev2-110                                                                                      | ⏳              | `use_transmission_map=True`<br>`use_gauss_filter=True`<br>`use_resize_conv=False`   |
| 2024-11-17-rev3-111                                                                                      | ⏳              | `use_transmission_map=True`<br>`use_gauss_filter=True`<br>`use_resize_conv=True`    |

</div>

### Results
The results of the new models are similar to the previous ones, here are some samples:
<div align="center">

| Clear                                                   | 2024-11-17-rev1-000                                        | 2024-11-17-rev2-110                                        | 2024-11-17-rev3-111                                        |
|---------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| <img src="images/results/2024-11-17/clear/sample1.jpg"> | <img src="images/results/2024-11-17/rev1-000/sample1.gif"> | <img src="images/results/2024-11-17/rev2-110/sample1.gif"> | <img src="images/results/2024-11-17/rev3-111/sample1.gif"> |
| <img src="images/results/2024-11-17/clear/sample2.jpg"> | <img src="images/results/2024-11-17/rev1-000/sample2.gif"> | <img src="images/results/2024-11-17/rev2-110/sample2.gif"> | <img src="images/results/2024-11-17/rev3-111/sample2.gif"> |
| <img src="images/results/2024-11-17/clear/sample3.jpg"> | <img src="images/results/2024-11-17/rev1-000/sample3.gif"> | <img src="images/results/2024-11-17/rev2-110/sample3.gif"> | <img src="images/results/2024-11-17/rev3-111/sample3.gif"> |

</div>
