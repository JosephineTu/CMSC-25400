Project 1 Report
Xiaoran Tu


1A.1

ridge_loss_function( ) 

![ridge_regression_function](ridge_regression_func.png) 

Description: 

1) define MSE as the mean of squared residuals: 
$MSE = {\frac{1}{n}}\sum_{i=1}^n(y_i - \mathbf{w}^{\top}\mathbf{x_i})^2$ 
2) define ridge loss as MSE + L2 penalty 
3) compute ridge gradient: 
$\nabla\hat{R}_{Ridge}({\mathbf{w}})={\frac{2}{n}}\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{y})+2\lambda\mathbf{w}$
 
1A.2 

gradient_descent_ridge( ) 

![ridge_gradient_descent](Ridge_gradient_descent.png) 

Description: 

1) compute ridge loss and gradient from the previous function 
2) compute the difference between the previous loss and the current round loss 
3) terminate when either iteration exceeds max_iter, or when difference is smaller than tolerance threshold 
4) update ${\hat{\mathbf{w}}}$ according to the learning rate and gradient: 
$$ \hat{\mathbf{w}}_{k+1} \gets \hat{\mathbf{w}}_k - \eta_k \left.\frac{\partial \hat{R}}{\partial \mathbf{w}} \right|_{\mathbf{w}=\mathbf{w}_k}$$

1A.3 

Test results

![test1A](test1A.png) 

1A.4 
true_gen_error( ) 

![tru_gen](true_GenE.png) 

Description:  

1) Expected true error. 

fit_naive_linear_regression( ) 

![fit_naive](fit_naive.png) 

Description: 

1) Closed form solution for converged gradient descent. 

Results and Questions  

![output5](output5.png) 

1) Higher observation noise level leads to higher expected true error and higher test error estimated from cross validation. The error estimated by CV is closest to true error, when the number of training samples around 15. 

1B.1 

train_and_eval_lambda( ) 

![train_and_eval_lambda](train_and_eval_lambda.png) 

Description: 

1) run gradient descent on train data, get the mse of the final round 
2) evaluate $\hat{\mathbf{w}}$ on the test dataset, compute test MSE loss 

![lamda_list_train_and_eval_1](lambda_list_1.png) 

Description: 

1) run train_and_eval_lambda( ) on different lambda values 
2) find the best lambda according to the smallest test_mse 

Analysis of Experimental Results 

![output_2](output2.png) 

1) The train MSE increases slowly at smaller $\lambda$ values. Once $\lambda$ exceeds 10^-1, the train MSE increases quickly. The test MSE exhibits a small decrease and global minima around 10^-2, then it increases together with the train MSE. 
2) Best $\lambda$ is 0.03. Regularization makes gradient descent converge at smaller $\hat{\mathbf{w}}$. This reduces overfitting over train data, making model more resilient to turbulence in dataset. The penalty term ${\lambda}{\mathbf{w}^\top\mathbf{w}}$ encourages smaller $\mathbf{w}$ by giving a weight decay factor of ($1-2\eta_t\lambda$) at every step.  
3) In theory, regularization should reduce variance and lead to a lower test MSE at an intermediate value of $\lambda$. In practice, we do not observe a strong improvement in test MSE as $\lambda$ increases. While the train MSE increases as expected, test MSE only shows a weak minimum. This might be due to the relative size of the train dataset (n=16512) and the actual data (20640). As we are sampling 80% of the actual data, the empirical risk is very close to the true risk, and the estimated parameters should have rather low variance, and the gain by regularization should be limited. When the biased term added on by regularization is large, the MSE of test data will increase rapidly.  

1B.2  

cross_validation_lambda( ) 

![cross_validation_lambda](cross_validation_lambda.png) 

Description: 

1) call the random number generator to generate a random sequance of indices (0, n) 
2) split the data into k folds, use the ${i^{th}}$ set folds[i] for testing, and the rest for training 
3) repeat this process for k iterations 
4) evaluate on train and test loss, return the mean MSE 

![kfold_cv](kfold_cv.png) 

Description: 

1) compute mean MSE from kfold cross validation 
2) find the best-fit $\lambda$ with the smallest mean test MSE 

Analysis of Experimental Results 

![cv_lambda](cv_lambda.png) 

best lambda: 10^-6 

1) The naive approach is to select the value of $\lambda$ that minimized the training MSE, since only training data is accessible.  
2) It reduces the variance in sampling validation data by averaging on the MSE across k=10 splits. Each data point is used for both training and testing, so the model is less sensitive to influence of an outlier or particular split. The single-validation approach might be overfitting to a validation dataset that is an unrepresentative sampling of the whole. 

1B.3 

train_and_eval_degree( ) 

![train_and_eval_degree](train_and_eval_degree.png) 

Description: 

1) safety reshape for X_train and X_eval 
2) stack X_train into a matrix, such that $x_{ij} = {x_i}^{j}$: 

$$
\begin{pmatrix}
x_1 & x_1^{2} & ... & x_1^{d}\\
x_2 & x_2^{2} & ... & x_2^{d}\\
...\\
x_n & x_n^{2} & ... & x_n^{d}
\end{pmatrix}
$$
3) do the same thing for X_eval 
4) call the train_and_eval_lambda( ) function from previous task 

cross_validation_degree( ) 

![cross_validation_degree](cross_validation_degree.png) 

Description: 

1) same as the previous k-fold cross validation cross_validation_lambda( ) 

Analysis of Experimental Results 

![output3](output3.png) 

Best degree: 4

1) The best degree is 4. This aligns with the minimum test error. 
2) When the polynomial degree m increases, the final train MSE is always going to decrease. This is because an (m-1)-degree polynomial in an m-1 dimension space $\mathbb{R}^{m-1}$ is always contained in the space $\mathbb{R}^{m}$. Therefore, we have $Train MSE_{d} \le Train MSE_{d+1}$. 
3) The zoomed-in plot shows that the actual difference in MSE across degrees is small, and the dataset is already well-explained by a polynomial model as small as degree 2. The alignment of CV MSE and test MSE shows that the model is quite stable to flucation in sampling test data, and is thus a good fit. This indicates that features encoded by a degree d>4 are largely irrelevant in this dataset.  

1C 

![lossfunc_lasso](lossfunc_lasso.png) 

Description: 

1) compute the loss function of Lasso regularization: $\mathcal{L} = {\frac{1}{n}}\sum_{i=1}^n\left(y_i-\theta^\top x_i\right)^2+{\lambda} ||\mathbf{\theta}||_1$ 

2) compute the gradient: 
Because $||\theta||$ is not differentiable at $\theta=0$, we separately differentiate it at $\theta<0$, $\theta=0$, and $\theta>0$ using np.sign( ).
$\nabla\hat{R}(\mathbf{\hat{\theta}}) = {\frac{2}{n}}(\mathbf{X}^\top(\mathbf{X}\mathbf{\theta}-\mathbf{y}))+\lambda\,\mathrm{sign}(\theta)$ 

![gradientdescent_lasso](gradientdescent_lasso.png) 

Description:  

1) same as previous gradient descent function gradient_descent_ridge( ) 

![train_and_eval_lasso](train_and_eval_lasso.png) 

Description: 

1) subtract the $\lambda||\theta||$ term from regLoss in lossFunctionLasso( ) to obtain the test MSE.  

Analysis of LASSO Regression Results 

![out4](out4.png) 

1) The optimal choice for $\lambda$ is 10^-8. The ridge regularization model have a weak decrease in test MSE at small $\lambda$ values, and then a sharp increase at a threshold value. The Lasso regularization model slow increase in test MSE at first, then it also sharply increases at a threshold, but then decreases steadily. 
While ridge shrinks the weight of every feature (entries in $\mathbf{w}$) smoothly, Lasso performs feature selection, and force many entires of $\mathbf{w}$ to be 0. Therefore, the optimal $\mathbf{\lambda}$ for them might be different.  