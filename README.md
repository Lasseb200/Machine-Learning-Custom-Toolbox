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
[Y_pred_idx, c] = km(X,k)

The function outouts a specific index $$\mathbf{Y}_{\text{pred,idx}}$$ for each data point corresponding to its predicted class. $$k$$ specifies the number of classes. Additionally, a confidence measure $$\mathbf{c}$$ of the classification being correct is provided.

### Naive-Bayes (NB)
[Y_pred, c, P, class_labels] = NB(X_training,Y_training,X_test)

The function outputs the predicted class label $$\mathbf{Y}_{\text{pred}}$$ for each test data point and the confidence in its classification $$\mathbf{c}$$. Additionally, the probability of other classes are included in the matrix $$\mathbf{P}$$, which columns corresponds to different classes. 'class_labels' contains the label for each column of $$\mathbf{P}$$.

### Bayesian Linear Discriminant Analysis (BLDA)
[Y_pred, c, P, class_labels] = BLDA(X_training,Y_training,X_test)

The function outputs the predicted class label $$\mathbf{Y}_{\text{pred}}$$ for each test data point and the confidence in its classification $$\mathbf{c}$$. Additionally, the probability of other classes are included in the matrix $$\mathbf{P}$$, which columns corresponds to different classes. 'class_labels' contains the label for each column of $$\mathbf{P}$$.

### Bayesian Quadratic Discriminant Analysis (QDA)
[Y_pred, c, P, class_labels] = QDA(X_training,Y_training,X_test)

The function outputs the predicted class label $$\mathbf{Y}_{\text{pred}}$$ for each test data point and the confidence in its classification $$\mathbf{c}$$. Additionally, the probability of other classes are included in the matrix $$\mathbf{P}$$, which columns corresponds to different classes. 'class_labels' contains the label for each column of $$\mathbf{P}$$.

### k-Nearest Neighbours (kNN)
[Y_pred, c, P, class_labels] = kNN(X_training,Y_training,X_test,f)

The function outputs the predicted class label $$\mathbf{Y}_{\text{pred}}$$ for each test data point and the confidence in its classification $$\mathbf{c}$$. $$f$$ specifies the number of nearest neighbours included. Additionally, the probability of other classes are included in the matrix $$\mathbf{P}$$, which columns corresponds to different classes. 'class_labels' contains the label for each column of $$\mathbf{P}$$.

### Support Vector Machine (SVM)
[Y_pred, c, P, class_labels] = SVM(X_training,Y_training,X_test,C,kernelType,kernelInput,multiclassModel,executionMode)

The function outputs the predicted class label $$\mathbf{Y}_{\text{pred}}$$ for each test data point and the confidence in its classification $$\mathbf{c}$$. Additionally, the probability of other classes are included in the matrix $$\mathbf{P}$$, which columns corresponds to different classes. 'class_labels' contains the label for each column of $$\mathbf{P}$$. Note that confidence and probability estimates require the "pairwiseCoupling" multiclass model.

$$C$$ specifies the punishment for margin violations. A large $$C$$ punishes margin violations heavier, while a smaller $$C$$ allow for more violations. 'kernelType' specifies the type of kernel used. Polynomial "poly", radial basis function "gaussian" and sigmoid "sigmoid" kernels are built-in, but other kernels can be added in "SVM\private\kernel_types.m". 'multiclassModel' specifies the method used for multiclass SVM. The function supports one-versus-rest "1vR", one-versus-one with hard voting "hard1v1", one-versus-one with soft voting "soft1v1" and pairwise coupling "pairwiseCoupling". 'executionMode' specifies whether the function is executed on a single core "single" or with parallel processing "parallel".
