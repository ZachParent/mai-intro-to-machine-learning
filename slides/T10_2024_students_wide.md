[Theory 10. Visualization](iml/slides/T10_2024_students_wide.pdf#page=2&selection=4,0,4,24)

[Visualizing low-dimensional views of high-dimensional data](iml/slides/T10_2024_students_wide.pdf#page=6&selection=7,0,11,16)

- mostly focused on
	- Self-organizing maps
	- multi-dimensional scaling

> ([p.12](iml/slides/T10_2024_students_wide.pdf#page=12&selection=6,0,12,23))
> Main goal is to communicate the information clearly and effectively through graphical means

![p.14](iml/slides/T10_2024_students_wide.pdf#page=14&rect=31,48,896,440)

[How do we visualize data of high (or even very high) dimensionality?](iml/slides/T10_2024_students_wide.pdf#page=15&selection=6,0,8,15)
- [Eliminate dimensions](iml/slides/T10_2024_students_wide.pdf#page=15&selection=12,0,12,20)
- [Divide & conquer](iml/slides/T10_2024_students_wide.pdf#page=15&selection=22,0,22,16)
- [Latent and projection models](iml/slides/T10_2024_students_wide.pdf#page=15&selection=29,0,29,28)

### Latency and Projection

> ([• Projection – Dimensionality compression • Multi-dimensional Scaling – Similitude information coding • Clustering – Finding grouping structure in data • E.g., Principal Components Analysis – Similitude information coding • Self-Organizing Map (SOM) & Generative Topographic Mapping (GTM) – They combine latent representation and clustering](iml/slides/T10_2024_students_wide.pdf#page=17))
> - **Projection**
> 	- Dimensionality compression
> 	- Multi-dimensional Scaling
> 	- Similitude information coding
> - **Clustering**
> 	- Finding grouping structure in data
> 		- E.g., Principal Components Analysis
> 	- Similitude information coding
> - **Self-Organizing Map (SOM) & Generative Topographic Mapping (GTM)**
> 	- They combine latent representation and clustering

#### Projection

> ([p 18](iml/slides/T10_2024_students_wide.pdf#page=18&selection=4,0,22,36))
> Representation in <4-D, so that the distance-neighborhood relations between multi-dimensional points are faithfully preserved
> - ==It is impossible== to preserve information integrally
> - Some scale normalization is required

![T10_2024_students_wide](iml/slides/T10_2024_students_wide.pdf#page=19&rect=54,160,647,426)

![T10_2024_students_wide](iml/slides/T10_2024_students_wide.pdf#page=20&rect=65,21,912,428)
#### Self-organizing maps (SOM)
SOM=self-organizing maps #acronym
https://youtu.be/0qtvb_Nx2tA
https://youtu.be/K4WuE7zlOZo

![T10_2024_students_wide](iml/slides/T10_2024_students_wide.pdf#page=23&rect=51,26,951,443)

[visualization and analysis tool for high dimensional data](iml/slides/T10_2024_students_wide.pdf#page=24&selection=7,0,10,0)

| ![T10_2024_students_wide](iml/slides/T10_2024_students_wide.pdf#page=25&rect=345,13,787,256)<br> | ![T10_2024_students_wide](iml/slides/T10_2024_students_wide.pdf#page=27&rect=508,6,834,232)<br> |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |

> ([p.29](iml/slides/T10_2024_students_wide.pdf#page=29))
> - During **training**, the “winner” neuron and its neighborhood adapts to make their weight vector more similar to the input pattern that caused the activation
> - The neurons are moved closer to the input pattern
> - The magnitude of the adaptation is controlled via a **learning parameter** which decays over time

##### Quantization Error

> ([p.30](iml/slides/T10_2024_students_wide.pdf#page=30))
> - Measures how well our neurons represent the input patterns
> 	- Note that there will be always some difference between the input pattern and the neuron it is mapped to
> - It is calculated by summing all the distances between each input pattern and the neuron to which is mapped.

![p.30](iml/slides/T10_2024_students_wide.pdf#page=30&rect=497,51,856,422)
##### Topological Error

> ([p.31](iml/slides/T10_2024_students_wide.pdf#page=31&selection=4,0,25,21))
> - Evaluates the complexity of the output space
> - Measures the number of times the second closest neighbor in the input space is not mapped into the neighbourhood of the neuron in the output space
> - A high topological error may indicate that the classification problem is complex or may suggest that the training was not adequate and the network is folded

![p.32](iml/slides/T10_2024_students_wide.pdf#page=32&rect=132,15,840,410)

![p.34](iml/slides/T10_2024_students_wide.pdf#page=34&rect=75,14,887,440)

![p.35](iml/slides/T10_2024_students_wide.pdf#page=35&rect=38,4,955,448)

#### [SOM algorithm](iml/slides/T10_2024_students_wide.pdf#page=36&selection=0,0,0,13)

==pay attention here==

## [Multi-Dimensional Scaling (MDS)](iml/slides/T10_2024_students_wide.pdf#page=49&selection=2,0,8,5)

> ([T10_2024_students_wide, p.50](iml/slides/T10_2024_students_wide.pdf#page=50&selection=18,0,24,41))
> The goal of an MDS analysis is to find a spatial configuration of objects when all that is known is some measure of their general (dis)similarity.

> ([p.51](iml/slides/T10_2024_students_wide.pdf#page=51&selection=2,0,30,52))
> - Generally regarded as **exploratory data analysis**
> - **Reduces large amounts of data** into easy-to-visualize structures
> - Attempts to **find structure** (visual representation) in a set of distance measures, e.g. dis/similarities, between objects/instances
> 	- Shows how variables/objects are related perceptually

> ([p.51](iml/slides/T10_2024_students_wide.pdf#page=51&selection=32,0,52,36))
> - How? By assigning instances to specific locations in space
> - Distances between points in space match dis/similarities as closely as possible:
> 	- Similar objects: Close points
> 	- Dissimilar objects: Far apart points

==learn the algorithm==

#### Output of MDS
> ([p.63](iml/slides/T10_2024_students_wide.pdf#page=63&selection=5,0,34,86))
> 1) Clusters: Groupings in a MDS spatial representation.
> 	- These may represent a domain/subdomain.
> 1) Dimensions: Hidden structures in data. Ordered groupings that explain similarity between items.
> 	- Axes are meaningless and orientation is arbitrary.
> 	- In theory, there is no limit to the number of dimensions.
> 	- In reality, the number of dimensions that can be perceived and interpreted is limited.

==learn advantages and disadvantages==

> ([p.66](iml/slides/T10_2024_students_wide.pdf#page=66&selection=27,0,78,17))
> Taxonomy:
> - **Metric multidimensional scaling** -- assumes the input matrix is just an item-item distance matrix. Analogous to PCA, an eigenvector problem is solved to find the locations that minimize distortions to the distance matrix. Its goal is to find a Euclidean distance approximating a given distance.
> - **Generalized multidimensional scaling (GMDS)** -- A superset of metric MDS that allows for the target distances to be non-Euclidean.
> - **Non-metric multidimensional scaling** -- It finds a non-parametric monotonic relationship between the dissimilarities in the item-item matrix and the Euclidean distance between items, and the location of each item in the low-dimensional space
