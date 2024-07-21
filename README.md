# CMRI-Insight
**CMRI Insight: A GUI-Based Open-Source Tool for Cardiac MRI Segmentation and Motion Tracking**

## Overview
CMRI Insight is an advanced Python-based Graphical User Interface (GUI) designed for automated analysis of cardiac MRI images. It integrates deep learning models for segmentation and motion tracking, providing functionalities for both automated and manual segmentation. This tool supports DICOM image localization and facilitates strain analysis, offering comprehensive visualization of cardiac structures.

## Features
- **Automated and Manual Segmentation**: Utilizes deep learning models for high-precision segmentation.
- **Dynamic 3D Mesh Generation**: Illustrates changing frames and sparse fields of the left ventricular myocardium.
- **Strain Analysis**: Calculates and analyzes circumferential and radial strains.
- **DICOM Image Support**: Provides position and orientation information.
- **Customizable Models**: Supports loading of custom localizer and segmentation models.

## Version 2
<div align="center">
  <div style="float:left;margin-right:10px;">
    <img src="https://github.com/Hamed-Aghapanah/CMRI-Insight/blob/main/images/version2.gif" width="900px"><br>
  </div>
  <div style="float:right;margin-right:0px;">
    <p style="font-size:3.5vw;">  Gif 1 : GUI Version 2 </p>
  </div>
</div>

## Version 1
<div align="center">
  <div style="float:left;margin-right:10px;">
    <img src="https://github.com/Hamed-Aghapanah/CMRI-Insight/blob/main/images/version1.gif" width="900px"><br>
  </div>
  <div style="float:right;margin-right:0px;">
    <p style="font-size:3.5vw;">  Gif 2 : GUI Version 1 </p>
  </div>
</div>

## Introduction

Briefly introduce the project and its purpose. Explain why cardiac localization with YOLO is relevant and provide a high-level overview of the implementation.
Requirements

List the software and hardware requirements necessary to run the code. Include specific versions if applicable.

    Python 3.x
    CUDA and cuDNN (if using GPU)
    Additional dependencies (TensorFlow 2+, Keras, PyTorch, Matplotlib, pandas, NumPy, SciPy, PyQt 5, OpenCV)

## Installation

```shell
git clone https://github.com/Hamed-Aghapanah/CMRI-Insight.git
# Change into the project directory
cd CMRI-Insight
# Install dependencies
pip install -r requirements.txt
python CMRI-Insight_GUI.py 
```



## Flowchart of CMRI Insight
<div align="center">
<img src="https://github.com/Hamed-Aghapanah/CMRI-Insight/blob/main/images/Graphical%20Abstract.png" width="600px" alt="Flowchart of CMRI Insight GUI">
    <p>Fig. 1 Flowchart of CMRI Insight GUI </p>

</div>


## Example of Localization
### Sample CMRI Image Displaying Localization Outcomes

<div align="center">
<img src="https://github.com/Hamed-Aghapanah/CMRI-Insight/blob/main/images/croped_test_batch1_pred.jpg" width="600px" alt="Fig. 3 Array of cardiac MRI images showcasing the application of YOLOv7 technology for precise region of interest (ROI) identification, highlighting the automated detection and localization capabilities within various cardiac structures">

  <p>Fig. 2 Array of cardiac MRI images showcasing the application of YOLOv7 technology for precise region of interest (ROI) identification, highlighting the automated detection and localization capabilities within various cardiac structures </p>
</div>


## Example of Segmentation
### Sample CMRI Image Displaying Segmentation Outcomes
<div align="center">
<img src="https://github.com/Hamed-Aghapanah/CMRI-Insight/blob/main/images/00.PNG" width="600px" alt="CMRI Segmentation Example">
   <p>Fig.3 CMRI Segmentation Example.</p>
</div>

This interface features multiple views of the heart with segmented regions distinctly highlighted in various colors:

<div align="left">
<img src="https://github.com/Hamed-Aghapanah/CMRI-Insight/blob/main/images/legend.JPG" width="200px" >
</div>


Additionally, it provides comprehensive patient information, DICOM data, and navigation controls for examining MRI slices.

## 3D Visualization
### 3D Mesh and Bull's Eye Pattern
<div align="center">
  <img src="https://github.com/Hamed-Aghapanah/CMRI-Insight/blob/main/images/5555.PNG" width="900px" alt="3D Mesh Visualization">
  <p>Fig. 4: 3D Visualization of Contours Extracted from CVI42. Initial 3D mesh (A), recolored 3D mesh featuring a bull’s eye pattern (B), and bull’s eye pattern (C).</p>
</div>

## Tabs Introduction
The GUI is structured with multiple tabs for efficient navigation:
- **File**: Tools for project and file management.
- **Edit**: Data manipulation and management features.
- **Image**: Data extraction and enhancement functionalities.
- **Segmentation**: Manual and automatic segmentation options.
- **Analysis and Tracking**: Provides tools for analyzing and tracking cardiac motion and strain.
- **Tools**: Additional utilities for enhancing the analysis process, such as measurement tools and annotation features.
- **View**: Options for customizing the display and visualization settings of the images and analysis results.
- **Help**: Access to user guides, documentation, and support resources.



## Future Work
- **Improvement of User Interface**: Enhancing UI for better user experience and real-time collaboration.
- **Transition from MATLAB to Python**: Rewriting code for broader accessibility and advanced functionalities.

## Code Metadata
| Code Metadata                                | Description                                                                                 |
|----------------------------------------------|---------------------------------------------------------------------------------------------|
| **Current code version**                     | v2.1.4                                                                                      |
| **Permanent link to code/repository**        | [GitHub Repository](https://github.com/Hamed-Aghapanah/CMRI-Insight)                        |
| **Permanent link to reproducible capsule**   | [Code Ocean Capsule](https://codeocean.com/capsule/4020757/tree)                         |
| **Legal code license**                       | MIT License                                                                                 |
| **Code versioning system used**              | Git                                                                                         |
| **Software code languages, tools, services** | Python, GitHub                                                                              |
| **Compilation requirements**                 | Python 3.8+, TensorFlow 2+, Keras, PyTorch, Matplotlib, pandas, NumPy, SciPy, PyQt 5, OpenCV |
| **Support email**                            | [h.aghapanah@amt.mui.ac.ir](mailto:h.aghapanah@amt.mui.ac.ir)                               |

## Contact
For any questions or support, please contact:
- Hamed Aghapanah : [h.aghapanah@amt.mui.ac.ir](mailto:h.aghapanah@amt.mui.ac.ir)
- Saeed Kermani (Corresponding Author): [kermani@med.mui.ac.ir](mailto:kermani@med.mui.ac.ir)
- Reza Rasti    (Corresponding Author): [r.rasti@eng.ui.ac.ir](mailto:r.rasti@eng.ui.ac.ir)

