# Linear Regression

 - ***Absolute Error***, ***Squared Error*** and ***Huber Loss***
works when the assumption of the data being drawn from a ***line*** and for each data point, the label is drawn from a Gaussian.
![img](https://github.com/fuzailpalnak/ML-Scratch/tree/master/regression/linear/images/error.png)

- Mathematically the above diagram is formulated as following![equation](http://www.sciweavers.org/tex2img.php?eq=y%20%3D%20%20w%5E%7BT%7D%20%2A%20x%20%2B%20%20%5Cepsilon_%7Bi%7D%20%5C%5C%0Awhere%3B%20%5Cepsilon_%7Bi%7D%20%3D%20%20N%280%2C%20%5Csigma%20%5E2%29%0A%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
  
- The task of the Error is to estimate the slope ***w*** from the data that is derived from the ***Gaussian*** (see the equation and diagram above). 
If this model assumption about the data is violated, then the loss function will fail to achieve desired results
