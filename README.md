# Support-Vector-Machines-Pegasos
Pegasos is a version of SVM which stands for Primal Estimated sub-GrAdient SOlver. It is an effective stochastic gradient descent algorithm to solve a Support Vector Machine for binary classification problems. 

In this version, we reformulate SVM as an unconstrained minimization problem of empirical loss plus a regularization (L2) term (find the optimal w which minimizes the loss function). Pegasos performs stochastic gradient descent on this objective function, with learning rate decreasing with each iteration to guarantee convergence (to the optimum). 

