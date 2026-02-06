Project 1 Report

Xiaoran Tu

2A.1 Perceptron

![perceptron_1](project2/perceptron_1.png)
![perceptron_2](project2/perceptron_2.png)

Figure 1, Perceptron.fit( ):

1) set up error counter to trace the mistakes made by the model
2) Check if the label (y[j]) aligns with the decision function ($w^TX[j]+b$). If inconsistent, update w by $\eta * y[j] * X[j]$, and update b by $\eta * y[j]$.
3) exit loop if the model made no mistake throughout the training dataset

Figure 2, Perceptron.project( ):

1) calculate X_proj: $(Xw + b)$, the signed margin between the points and the learned hyperplane

Figure 2, Perceptron.predict( ):

1) use project( ), predict $\hat{y}$ based on the sign of X_proj

![output1_perceptron](project2/output1_perceptron.png)

Figure 3, Results

Questions:
1. Briefly discuss your implementation of the `Perceptron` algorithm. Looking at the visualization of the training set, do you expect it to converge? Hint: Justify using the perceptron convergence theorem.

[Answer]: The model should converge, because the training data are separable. According to the perceptron convergence theorem, if 

1) all inputs are bounded with $||x_i||\le R$; 
2) there exists a separating hyperplane with $y_i\hat{w}^Tx_i \ge \gamma > 0$, then the perceptron model makes at most $(\frac{R}{\gamma})^2$ mistakes. 

2A.2 Kernel Trick

![kerneltrick_1](project2/kerneltrick_1.png)

Figure 4, PolynomialKernel( ):

1) $k(x,x') = (x^Tx' + 1)^2$

Figure 4, GaussianKernel( ):

1) $k(x,x') = e^{\frac{-||x-x'||^2}{\sigma^2}}$

Figure 4, LaplaceKernel( ):

1) $k(x,x') = e^{\frac{-||x-x'||}{\sigma}}$

![kernel_fit](project2/kernel_fit.png)

Figure 5, KernelPerceptron.fit( ):

1) Define the $K_{ij}$ entry of the Gram matrix $K$ to be the inner product of $\phi(x)^T\phi(x)'$ through the kernel method $k(x[i],x[j])$.
2) $margin = \sum_{j=1}^n\left(\alpha_j y_j k(x_j, x\right))$, where $k(x_j, x_i)$ is the $K_{ji}$ entry of $K$. 
3) If $\hat{y} = sign(margin)$ is different from y, update alpha: $\alpha_j \leftarrow \alpha_j + 1 (\eta = 1)$ 

![kernel_project_predict](project2/kernel_project_predict.png)

Figure 6, KernelPerceptron.project( ):

1) Compute $\hat{y}$ in the test data with the trained $\alpha$:
$$\hat{y} = sign(\sum_{j=1}^n\left(\alpha_j y_j k(x_j, x\right)))$$

![degree_1_kernel](project2/degree_1_kernel.png)

Figure 7, Degree 1 kernel results
Accuracy: 47.50%

![Gaussian_kernel](project2/Gaussian_kernel.png)

Figure 8, GaussianKernel implementation 

![GaussianKernelacc100](project2/GaussianKernelacc100.png)

Figure 9, the test result of Gaussian Kernel under 100 training epochs
Accuracy: 97.50%
Decision boundaries are smooth and radial, points close in L2 distance are grouped. 

![GaussianKernel_epochs](project2/GaussianKernel_epochs.png)
![GaussianKernel_iters](project2/Gaussiankernel_iters.png)

Figure 10-11, Code and visualization for Gaussian Kernel accuracy in different training epochs. Approxiamtely 50 epochs is required for a plateau in accuracy.

![LaplacianKernel](project2/LaplacianKernel.png)

Figure 12, Laplace Kernel implementation

![Laplace](project2/laplace.png)

Figure 13, the test result of Laplacian Kernel

Accuracy: 97.50%
Decision boundaries are less smooth than Gaussian kernel and more diamond-shaped, points close in L1 distance are grouped. 

![Laplace_epochs](project2/Laplace_epochs.png)

Figure 14, Approximately 5 epochs is required for a plateau in accuracy for Laplacian kernel. 

![degree_2_polynomial](project2/degree_2_polynomial.png)

Figure 15, degree 2 polynomial implementation

![degree2polynomial](project2/degree2polynomial.png)

Figure 16, degree 2 polynomial kernel test results

Accuracy: 95.00%
Decision boundaries have a quadratic shape, because it computes $(x^T x'+1)^2$, which includes all degree-2 polynomial terms. 

![polynomialepochs](project2/polynomialepochs.png)

Figure 17, Approximately 20 epochs is required for a plateau in accuracy for degree 2 polynomial kernel. In larger number of epochs, the model began to overfit.

Questions:
1. Report on the performance and behavior of the linear kernel (polynomial with $p = 1$) model. Do you expect this performance?
    
[Answer:] The performance of around 50% accuracy (about the probability of random choice) is expected. This is because the data is not linearly separable, therefore a polynomial of degree 1 will not be able to outperform a random separation. 

2. **Briefly** discuss your implementation for the kernel perceptron. For each of the kernels (Gaussian, Laplace, and Polynomial) you implemented, report on their decision boundaries, accuracy, and how the number of epochs required to reach a plateau in accuracy. Analyze what you see qualitatively and quantitatively. 

[Answer:] 
1) (For all 3 kernels): find optimal $\alpha$ values based on the training set; evaluate on test set with the trained $\alpha$
2) [Copy of annotations under figures]:
*Gaussian Kernel:*
Accuracy: 97.50%
Decision boundaries are smooth and radial, points close in L2 distance are grouped. 
Approxiamtely 50 epochs is required for a plateau in accuracy.
*Laplace Kernel:*
Accuracy: 97.50%
Decision boundaries are less smooth than Gaussian kernel and more diamond-shaped, points close in L1 distance are grouped. 
Approximately 5 epochs is required for a plateau in accuracy for Laplacian kernel. 
*Degree 2 Polynomial kernel:*
Accuracy: 95.00%
Decision boundaries have a hyperbola shape. 
Approximately 20 epochs is required for a plateau in accuracy for degree 2 polynomial kernel. In larger number of epochs, the model began to overfit.

2B

![sigmoid_kernel](project2/sigmoid_kernel.png)

Figure 18, Sigmoid Kernel
1) $k_{Sigmoid}(x,x',\gamma,c) = tanh(\gamma x^T x' + c)$

![polynomial_candidates](project2/polynomial_kernel_new.png)

Figure 19, Find best p for the polynomial kernel:

The best p is 5, with test accuracy 79.00%.
1) Train the model based on polynomial kernels degree [1,2,3,4,5]
2) Find the best-performing polynomial degree based on the accuracy on validation set
3) Apply the degree-p polynomial kernel to test dataset, retrain and evaluate accuracy.

![gaussiankernel_sigma](project2/gaussiankernel_sigma.png)

Figure 20, Find best sigma for Gaussian kernel:

1) Normalize training, validation and test dataset on the mean and standard deviation of Xtrain.
2) Calculate pairwise distance between different rows of Xtrain: 
$$d_{ij} = \sqrt{\sum_{k=1}^d\left(x_{ik}-x_{jk}\right)^2}$$
3) Find the median of all pairwise distances, choose candidate sigmas on a logarithmic scale. 
4) Select the best sigma value based on validation set performance, and record its generalizability in the test set.

![Gaussian_output](project2/gaussianout_new.png)

Figure 21, best sigma value is 6.65, with a test accuracy of 79.00%

![sigmoidkernel_params](project2/sigmoidkernel_params.png)

Figure 22, find best $\gamma$ and c for Sigmoid kernel

1) Select candidate $\gamma$ to be in inverse ratio with the size of the dataset. Suppose that the input data x,y follows the Gaussian distribution $(x,y)\sim \mathcal{N} (0,1)$. Then we have: $$Var(x_iy_i)=E[x_i^2]E[y_i^2]=1$$ and $$Var(x^Ty)=\sum_{i=1}^nVar(x_iy_i)=n$$
this gives that $$z=\gamma x^Ty \sim O(\sqrt{Var(x^Ty)})=O(\gamma\sqrt{n})$$
Because the gradient of $tanh$ vanishes at $|z| \approx 1$, for a conservative choice we scale $\gamma$ with 1/n, and the offset parameter c within {-1, 0, 1} such that $\gamma x^Ty+c$ stays within the linear regime of $tanh$.
3) Repeat a similar process as selecting the best parameters for Gaussian kernel. 

![sigmakernel_result](project2/sigmoid_new.png)

Figure 23, best $\gamma$ value is 0.08, best c value is 1.0, with a test accuracy of 84.00%. 

![laplace_params](project2/laplace_params.png)

Figure 24, Find the best sigma for Laplace kernel

The best sigma is 0.2 with test accuracy 80.50%.

1) Use log-scale grid search parameters [0.01, 0.02, 0.05, 0.1, 0.2] for sigma
2) Repeat a similar process as selecting the best parameters for Gaussian and Sigmoid kernel. 

Questions:
1. Discuss how accuracy compares across all methods. What would be the best and worst kernels? And what are their accuracy scores **on test set**?

[Answer:] 
Best kernel: Sigmoid kernel, with test accuracy 84.00%.
Worst kernel: Laplace, with test accuracy 78.50%. 

2. Analyze the discrepancy between the model's performance on the training set versus the unseen test set (i.e., evaluate the generalization gap). Define overfitting in terms of these metrics.

[Answer:] 
All models have a mild discrepancy between train and test error (around 1-3%). 
1) The overfitting of Polynomial kernel might happen in high degrees.
2) Gaussian kernel can overfit with small $\sigma$ values, leading to highly localized decision boundaries and large variation. 
3) Sigmoid model can overfit under large $\gamma$ values, when tanh saturates.
4) Laplace kernel is a rather simple model, and has a relatively lower accuracy due to bias.
In general, because the model performances are evaluated on validation sets, the generalization gap should be rather small comparing to optimizing on training sets.


2C.1

![acc_cv](project2/acc_cv.png)

Figure 25, Implementation of cross validation using accuracy

1) Split Xtrain and ytrain into train and validation set. 
2) Fit the model on training set, and evaluate its accuracy on validation set.
3) return the mean accuracy

![greedyforward](project2/greedyforward.png)

Figure 26, Implementation greedy forward feature selection 

1) Create a mask on the all the features $V$ in X.
2) For all the remaining feature ${j} \in {V\setminus S}$, find the best feature $s_i$ to add, such that
$$s_i = \operatorname{argmax}_{j \in \{V\setminus S\}} {A_{cv}(S \cup \{j\})}$$
3) Stop early if the max accuracy with $s_i$ is lower than the previous accuracy.
4) update $S$: $S\leftarrow \{S\cup s_i \}$, set the mask value to True. 

![greedyoutput](project2/greedyoutput.png)

Figure 27, The chosen features are: 

Index['Temperature (deg C)', 'Hour', 'Rainfall(mm)' ,'Solar Radiation (MJ/m2)']

![greedyresult](project2/greedyresult.png)

Figure 28, Accuracy of training the selected features on test set is 83.50%.

1) Use the mask to extract the 4 chosen features.

2C.2

![SVM_lossfunc](project2/SVM_lossfunc.png)

Figure 29, L1-SVM Loss function

1) Create a mask for all the misclassified or small margin datapoints, where $1-y_iw^Tx_i > 0$.
2) The gradients at these points is $(-\frac{1}{n})(x^Ty)+\lambda sign(w)$. Otherwise, it is $\lambda sign(w)$.
3) The loss is given by:
$$
    L = \frac{1}{n} \sum_{i=1}^n \max(0, 1 - y_i \mathbf{w}^T \mathbf{x}_i) + \lambda \lVert w \rVert_1 
$$

![gradient_svm](project2/gradient_svm.png)

Figure 30, L1-SVM gradient descent

1) Run l1_svm_loss_function( ) to obtain the loss and gradient. 
2) If the difference between 2 consecutive losses is smaller than the tolerance value, stop early. 
3) Otherwise, update w: 
$$w\leftarrow w-\eta \nabla L$$
4) Update loss_history

![output](project2/output.png)

Figure 31, L1-SVM performance with respect to the regularization coeffient $\lambda$

Best $\lambda$ value: 0.001

1. Let's say our lambda list is fixed, and hyperparameters (such as k, num_iter, tolerance, etc.) are fixed. You don't have to take into consideration the plot above. Why might there be an issue with the lambda selection procedure above?
   
[Answer:] 
Because the features is selected across the whole dataset before performing CV, it is optimal and therefore biased. This causes better performance in cross validation, and a biased choice of $\lambda$. 