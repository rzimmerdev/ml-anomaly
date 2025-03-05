# ml-anomaly
This is the source code for the paper Machine Learning Techniques for Improving Multiclass Anomaly Detection. Provides an improvement on the existing dataset analysis paper Evaluating Conveyor Belt Health With Signal Processing Applied to Inertial Sensing

## Installation
```bash
conda create -n ml-anomaly --file requirements.txt python=3.10
```

## Usage
```bash
conda activate ml-anomaly
python test.py
```

Industrial conveyor belt systems are an efficient means of transport due to their adaptability and extension. Nonetheless, such systems are prone to various failures, including but not limited to: idler anomalies; belt tears; and pin misalignment which can cause significant disruptions in the production process. Preemptive maintenance and health monitoring of these conveyor belts is a common practice for avoiding these failures, but a challenging task due to the rarity of comprehensive anomaly detection datasets in the area, with current works aimed at evaluating the belt's immediate condition at fixed points. This study addresses this research gap by comparing multiple machine learning techniques, such as a proposed Hybrid Neural Network (HNN) tailored for classification of multiple anomaly classes, as well as machine learning approaches for time series based on feature extraction, Catch22, Minirocket Arsenal, and Time Series Forest.

![image](https://github.com/rzimmerdev/ml-anomaly/blob/main/results.png)

Models based on feature extraction performed well in terms of accuracy, especially ensemble strategies such as MultiRocket and MiniRocket Arsenal. Of particular note is the MiniRocket, which sim than the HNN and MultiRocket. The others provided inferior results, but they are simpler models that don't use the ensemble strategy or even extract features like RF. One observation is that catch22, a feature extractor, performed worse than RF without feature extraction. However, the Time Series Forest algorithm, which extracts features from sub sequences, proved to be a more interesting way of classifying our time series, with better results than RF, classifying the series with features extracted from the whole series with catch22 and also RF classifying the series with the raw data. 

In summary, our Hybrid Transformer model's performance demonstrated its potential as a robust tool for predictive maintenance and anomaly detection, demonstrating a marked advancement over existing anomaly detection methods.

