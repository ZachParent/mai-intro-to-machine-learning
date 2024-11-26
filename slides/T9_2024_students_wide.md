# Factor Analysis
There is controversy of whether PCA is factor analysis or not
![T9_2024_students_wide, p.6](T9_2024_students_wide.pdf#page=6&rect=406,192,909,405)
> [!PDF|] [T9_2024_students_wide, p.7](T9_2024_students_wide.pdf#page=7&selection=2,0,73,31)
> > “Perhaps the most widely used (and misused) multivariate [technique] is factor analysis. Few statisticians are neutral about this technique. Proponents feel that factor analysis is the greatest invention since the double bed, while its detractors feel it is a useless procedure that can be used to support nearly any desired interpretation of the data. The truth, as is usually the case, lies somewhere in between. Used properly, factor analysis can yield much useful information; when applied blindly, without regard for its limitations, it is about as useful and informative as Tarot cards. In particular, factor analysis can be used to explore the data for patterns, confirm our hypotheses, or reduce the many variables to a more manageable number. -- Norman Streiner, PDQ Statistics

> [!PDF|] [T9_2024_students_wide, p.8](T9_2024_students_wide.pdf#page=8&selection=12,0,12,28)
> > To explore data for patterns
>

> [!PDF|] [T9_2024_students_wide, p.10](T9_2024_students_wide.pdf#page=10&selection=2,0,6,59)
> > GOAL To summarize patterns of correlations among observed variables
> 
>

![T9_2024_students_wide, p.13](T9_2024_students_wide.pdf#page=13&rect=190,91,807,194)

$X$ is a particular instance

translations can be considered as a part of the transformation too. doesn't change the data, but shifts it positively or negatively

![T9_2024_students_wide, p.14](T9_2024_students_wide.pdf#page=14&rect=47,13,866,438)

> [!PDF|] [T9_2024_students_wide, p.18](T9_2024_students_wide.pdf#page=18&selection=4,0,6,15)
> > Confirmatory Factor Analysis
> 
>

![](T9_2024_students_wide.png)

[T9_2024_students_wide, p.22](T9_2024_students_wide.pdf#page=22&rect=147,54,875,422)

> [!PDF|yellow] [T9_2024_students_wide, p.24](T9_2024_students_wide.pdf#page=24&selection=18,0,24,32&color=yellow)
> > Goal of PCA: find new representation (basis) to filter the noise and reveal hidden dynamics

PCA is the most common form of factor analysis

> [!PDF|yellow] [T9_2024_students_wide, p.25](T9_2024_students_wide.pdf#page=25&selection=15,0,33,31&color=yellow)
> > The new variables/dimensions
> > - Are linear combinations of the original ones
> > - Are uncorrelated with one another
> > 	- Orthogonal in original dimension space
> > - Capture as much of the original variance in the data as possible
> > - Are called Principal Components

![T9_2024_students_wide, p.28](T9_2024_students_wide.pdf#page=28&rect=317,128,698,425)

> [!PDF|] [T9_2024_students_wide, p.28](T9_2024_students_wide.pdf#page=28&selection=16,0,16,67)
> > Projections along PC1 discriminate the data most along any one axis

Covariance matrix will be square no matter how many features or samples

![T9_2024_students_wide, p.36](T9_2024_students_wide.pdf#page=36&rect=624,114,891,404)

$Au = \lambda u$, where $u$ is the eigenvector, $A$ is a matrix, $\lambda$ is a scalar, called the eigenvalue

![T9_2024_students_wide, p.39](T9_2024_students_wide.pdf#page=39&rect=55,12,926,445)

![T9_2024_students_wide, p.40](T9_2024_students_wide.pdf#page=40&rect=326,37,706,386)

## Independent Component Analysis (ICA)

![T9_2024_students_wide, p.58](T9_2024_students_wide.pdf#page=58&rect=180,57,814,401)

> [!PDF|] [T9_2024_students_wide, p.64](T9_2024_students_wide.pdf#page=64)
> > “Independent component analysis (ICA) is a method for finding underlying factors or components from multivariate (multi-dimensional) statistical data. What distinguishes ICA from other methods is that it looks for components that are both statistically independent, and nonGaussian.” A.Hyvarinen, A.Karhunen, E.Oja ‘Independent Component Analysis’ What is ICA?
> 
>

![T9_2024_students_wide, p.65](T9_2024_students_wide.pdf#page=65&rect=174,7,542,440)

> [!PDF|] [T9_2024_students_wide, p.69](T9_2024_students_wide.pdf#page=69&selection=2,0,32,37)
> > * Central Limit Theorem
> > 	* If two random (non-Gaussian) signals are added, the resulting signal will be more Gaussian than the original two random signals
> > * ICA Separation Concept 
> > 	* Central Limit Theorem (in Reverse)
> > 	* Maximizing Non-Gaussianity • Results in separating the two signals

![T9_2024_students_wide, p.83](T9_2024_students_wide.pdf#page=83&rect=207,9,772,440)