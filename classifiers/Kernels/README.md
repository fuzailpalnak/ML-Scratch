# Kernelized Algorithm

Kernelizing an algorithm is a two step process.
1. To show Data is only accessed in terms of inner products
2. Substitute the inner products with the kernel function

If the classifier access everything in terms of inner product than the inner products can be replaced by a kernelized 
version, for classification, the kernelized value of *_w.T * x* is 
 
![CodeCogsEqn (6)](https://user-images.githubusercontent.com/24665570/90312601-5fa94680-df23-11ea-9244-d8fa400b6cd3.png)

The equation for the weight vector *_w_* is derived from Dual SVM problem.
