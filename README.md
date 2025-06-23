# Kalman-Enhanced Streaming Linear Discriminant Analysis for Land Use Classification in Satellite Imagery

Satellite Image Classification in a streaming context presents several challenges that must be addressed to ensure accuracy, efficiency, and adaptability.

<p align="center"> <img src="Images/various/crop field rigoglioso.png" alt="Example of a crop field with lush vegetation" width="45%"> <img src="Images/various/crop field secco.png" alt="The same crop field with dry vegetation" width="45%"> </p>
Figure: Illustration of seasonal variations in satellite images. The left image shows a crop field with lush vegetation, while the right image shows the same field during a dry period.

A significant challenge in learning from data streams is that they are typically temporally arranged, with changes in the distribution resulting from the passage of time. In the context of time-ordered image streams, the temporal variation in the data distribution is formally defined as temporal distribution shift. Land-use categories, such as agricultural fields or urban areas, may appear differently over time due to seasonal changes or urbanization, while still representing the same class. These temporal variations can complicate the classification process, requiring models that are capable of adapting to evolving patterns in the data, even when the labels remain static. The figure above illustrates an example of seasonal effects, where two images of the same crop field exhibit contrasting vegetation states.

Additionally, the high dimensionality of satellite images, which often contain vast amounts of information spread across multiple channels, must be taken into account. Processing and analyzing such large-scale data in real time is computationally expensive, requiring the use of dimensionality reduction techniques to reduce processing costs without losing critical information.
Although dimensionality reduction algorithms are commonly employed to simplify data before classification, selecting the most effective approach is crucial. It requires determining the extent to which dimensionality can be reduced while still preserving the essential features necessary for accurate classification. Developing methods that incorporate data similarity while reducing dimensions is particularly important in maintaining classification quality.

Finally, achieving a balance between computational efficiency and classification accuracy is critical in streaming environments. High-dimensional data and the need for frequent model updates can increase processing times, which may not be feasible in real-time applications. Optimizing this trade-off between responsiveness and accuracy remains a central challenge.

This repository contains the tools and resources used for conducting simulations and analyses related to the FMoW-time dataset, a satellite image dataset containing more than 100'000 samples to be classified into more than 60 categories. Due to its large size, the FMoW-time dataset is not included in this repository. Instead, instructions and tools for creating the dataset locally are provided in the `Dataset` folder.

---

## Steps to Get Started

### Step 1: Install Dependencies
Ensure that all required dependencies are installed. Run the following command in your terminal:

```bash

pip install -r requirements.txt

```

### Step 2: Explore the Dataset
Dataset's content can be explored using the `Exploration.ipynb` notebook. This notebook provides an overview of the dataset structure and key insights.

### Step 3: Simulate a Pipeline
To execute the desired pipeline, use the `Complete simulation.ipynb` notebook. This notebook includes all necessary functions, which are specified in its preamble, to facilitate pipeline execution and saves extracted features in the `features` folder, relevant images in `figures` folder and relevant data in `saved_data` folder.

### Step 4: Visualize Simulation Results
For visual representation of the pipeline's results, refer to the `Plot results.ipynb` notebook.

### Step 5: Compare Simulations
Compare the performance of different pipelines in terms of execution time and classification accuracy using the `Comparing results.ipynb` notebook. The required comparison functions are included in the notebook's preamble.

## Additional Notes

### Supporting Files
The repository includes additional Python files that are required to support the execution of the provided notebooks. Ensure that these files are kept in their respective locations for smooth operation.
