# Linear Classifier

### Perceptron
- Perceptron assumes that the data is linearly seperable and if this assumption hold then 
perceptron will find the seperating hyperplane

### Logistic Regression
- Logistic Regression is often referred to as the discriminative counterpart of Naive Bayes. 
Here, ***P(y|xi)*** modeled and assume that it takes on exactly this form <br /> 
![equation](https://github.com/fuzailpalnak/ML-Scratch/blob/master/classifiers/linear/images/linear.png)
  
### Hinge
- It is often the case that there is no separating hyperplane between the two classes. 
In this case, there is no solution to the optimization problems stated above. 
We can fix this by allowing the constraints to be violated ever so slight with the introduction of slack variables
- The slack variable allows the input to be closer to the hyperplane or even be on the wrong side

