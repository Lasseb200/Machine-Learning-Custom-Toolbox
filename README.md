# Custom Toolbox for Machine Learning in Matlab
This toolbox features different dimensionality reduction and classification functions.
### Installation
The entire repository can be downloaded as a zip file and unpacked, but it is also possible to only download folders for the relevant functions. Functions can either be added to the matlab paths via 'HOME -> Set Path -> Add with Subfolders' or they can be loaded inside any Matlab document using 'load('Path...\Machine-Learning-Custom-Toolbox')'
### Data Format
Data must be provided in the format of an $$n\times d$$ data matix $$\mathbf{X}$$, where the $$n$$ rows corresponds to data points/experiments and the $$d$$ columns correspond to different features. Additionally, for supervised learners, the training class labels must be provided as a vector $$\mathbf{y}$$ with $$n$$ entries, all formatted as strings.

$$
\begin{align}
    \mathbf{X} = \begin{bmatrix}
        x_{11} & x_{12} & \dots & x_{1d}\\
        x_{21} & x_{22} & \dots & x_{2d}\\
        \vdots & \vdots & \ddots & \vdots\\
        x_{n1} & x_{n2} & \dots & x_{nd}
    \end{bmatrix}, \qquad \mathbf{y} = \begin{bmatrix}
        \text{"Class}_1\text{"} \\
        \text{"Class}_2\text{"} \\
        \vdots \\
        \text{"Class}_n\text{"}
    \end{bmatrix}
\end{align}
$$

## Dimensionality Reduction Methods
### Principal Component Analysis (PCA)
[v, lambda] = PCA(X)

The function outputs the principal components $$v$$ and corresponding eigenvalues $$\lambda$$. The dataset can be projected along the principal components using 'X_proj = X*v'.

### Linear Discriminant Analysis (LDA)
[v, lambda] = LDA(X_training, Y_training)

The function outputs the discriminant vectors $$v$$ and the corresponding eigenvalues $$\lambda$$. The dataset can be projected along the discriminant vectors using 'X_proj = X*v'. Additionally, the function includes regularisation of the within-class scatter-matrix.

### Orthogonal Linear Discriminant Analysis (OLDA)
[v, lambda] = OLDA(X_training, Y_training, n)

The function outputs the discriminant vectors $$v$$ and the corresponding eigenvalues $$\lambda$$. $$n$$ specifies the number of iterations/number of orthogonal discriminant vectors solved for. The dataset can be projected along the discriminant vectors using 'X_proj = X*v'. Additionally, the function includes regularisation of the within-class scatter-matrix.

## Classification Methods
### k-Means (km)
[Y_pred_idx, confidence] = km(X,k)

The function outouts an index $$Y_{pred,idx}$$ 
    

### Naive-Bayes (NB)
### Bayesian Linear Discriminant Analysis (BLDA)
### Bayesian Quadratic Discriminant Analysis (QDA)
### k-Nearest Neighbours (kNN)
### Support Vector Machine (SVM)
