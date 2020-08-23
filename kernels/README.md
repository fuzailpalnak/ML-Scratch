# Kernels

A linear classifiers can be made non-linear by applying basis function (feature transformations)
on the input feature vectors, i.e apply transformation x→ϕ(x).

![CodeCogsEqn (2)](https://user-images.githubusercontent.com/24665570/90312309-0b9d6280-df21-11ea-867b-7db98eb3af73.png)

This new representation, ϕ(x), is very expressive and allows for complicated non-linear decision boundaries.


The kernel trick is a way to get around this dilemma by learning a function in the much higher dimensional space, 
without ever computing a single vector ϕ(x) or ever computing the full vector *w* for that matter.

![CodeCogsEqn (3)](https://user-images.githubusercontent.com/24665570/90312330-2e2f7b80-df21-11ea-9365-3bfb1cbbd029.png)
![CodeCogsEqn (4)](https://user-images.githubusercontent.com/24665570/90312345-5b7c2980-df21-11ea-84bd-0c5f192d6cb4.png)

## Radial Bias Function Kernel (RBF)
There are several kernel functions present out of which Radial Bias Function Kernel is popular.
The RBF Kernel is defined as:-
![CodeCogsEqn (5)](https://user-images.githubusercontent.com/24665570/90312462-481d8e00-df22-11ea-8a77-a37e8234c92a.png)

## Polynomial kernel
![CodeCogsEqn (3)](https://user-images.githubusercontent.com/24665570/90977252-a3becb80-e561-11ea-9ade-4926b61fa935.png)
