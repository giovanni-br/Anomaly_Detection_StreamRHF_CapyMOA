# **StreamRHF in [CapyMOA](https://capymoa.org)**

This repository contains an implementation of the [**STREAMRHF algorithm**](https://inria.hal.science/hal-03948938/document), a tree-based unsupervised anomaly detection model specifically designed for **data streams**. STREAMRHF has been integrated into the open-source Python library [**CapyMOA**](https://capymoa.org), allowing it to effectively handle real-time anomaly detection tasks. 

---

## **Setup Instructions**

### 1. Install CapyMOA
To get started, follow the **CapyMOA developer installation guide**:  
[**CapyMOA Installation Guide**](https://capymoa.org/installation#install-capymoa-for-development)
Instead of cloning the original CapyMOA repository, clone this repository to get the customized implementation of STREAMRHF.
### 2. Clone This Repository
Clone this repository to your local environment:
```bash
git clone https://github.com/AlejandroUN/Stream-Random-Histogram-Forest
cd Stream-Random-Histogram-Forest
```

## **Implementation of the code and Demo**
The from-scratch Python implementation of the StreamRHF algorithm is located at:
```bash
cd  Stream-Random-Histogram-Forest/src/capymoa/anomaly/_stream_rhf.py
```
The demo, which integrates the StreamRHF implementation with CapyMOA and includes model comparisons and plots, is available in the following notebook:
```bash
cd Stream-Random-Histogram-Forest/src/capymoa/anomaly/streamrhf/notebook_presentation.ipynb

```

## **Contact**
For any questions, feel free to reach out to:

* giovanni.benedetti-da-rosa@polytechnique.edu
* cristian.chavez-becerra@polytechnique.edu


