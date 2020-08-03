# Linear Regression

 - ***Absolute Error***, ***Squared Error*** and ***Huber Loss***
works when the data is being drawn from a ***line*** and for each data point, the label is drawn from the line with some error ***epsilon***.
This Model Assumption has to hold, while considering these loss function.
![img](https://github.com/fuzailpalnak/ML-Scratch/blob/master/regression/linear/images/error.png)

- Mathematically the above diagram is formulated as follows <br /> ![equation](https://github.com/fuzailpalnak/ML-Scratch/blob/master/regression/linear/images/eqn1.png)
  
- The task of the Error is to estimate the slope ***w*** from the data that is derived from the ***Gaussian*** (see the equation and diagram above). 
If this model assumption about the data is violated, then the loss function will fail to achieve desired results
