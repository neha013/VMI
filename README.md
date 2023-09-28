# Visual-Motion-Interaction-Guided-Pedestrian-Intention-Prediction-Framework
![image](https://user-images.githubusercontent.com/41139808/222060458-2e3d6fd6-ccfc-4cbe-a6ef-fa5e8511fd7b.png)

This is the official python implementation for paper ***Neha Sharma, Chhavi Dhiman, S. Indu***, [*"Visual-Motion-Interaction Guided Pedestrian Intention Prediction Framework".*](https://ieeexplore.ieee.org/document/10264803)
 
The capability to comprehend the intention of
pedestrians on road is one of the most crucial skills that the current
autonomous vehicles (AVs) are striving for, to become fully
autonomous. In recent years, various multimodal methods have
become popular in predicting pedestrian crossing intention that
utilizes information from different modalities like trajectory,
appearance, context, etc. However, most existing research works
still lag rich feature representational ability in a multimodal
scenario, restricting the performance of these works. Moreover,
less emphasis is put on pedestrian interactions with the
surroundings for predicting short-term pedestrian intention in a
challenging ego-centric vision. To address these challenges, an
efficient Visual Motion Interaction guided intention prediction
framework has been proposed in this work. This framework
combines three divisions namely, Visual Encoder (VE), Motion
Encoder (ME) and Interaction Encoder (IE) to capture rich
multimodal features of the pedestrian and its interactions with the
surroundings, followed by temporal attention and adaptive fusion
module to integrate these multimodal features efficiently. The
proposed framework outperforms several SOTAs on benchmark
datasets: PIE/JAAD with Accuracy, AUC, F1-score, Precision and
Recall as 0.92/0.89, 0.91/0.90, 0.87/0.81, 0.86/0.79, 0.88/0.83
respectively. Furthermore, extensive experiments are carried out
to investigate the effect of different fusion architectures and design
parameters of all encoders. The proposed VMI framework is able
to predict the pedestrian crossing intention 2.5 sec ahead of the
crossing event. 


## Datasets
The proposed framework is trained and tested on the benchmark [PIE](http://data.nvision2.eecs.yorku.ca/PIE_dataset/) and [JAAD](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) datasets. The precomputed pose features for both the datasets are available inside folder `data/features/pie/poses/` and `data/features/jaad/poses/`. 
