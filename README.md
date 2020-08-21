***
# Matlab Code for Scalable Graph Hashing with Feature Transformation
***
## introduction
* This package contains the source code for the following paper:
	* Qing-Yuan Jiang and Wu-Jun Li. **Scalable Graph Hashing with Feature Transformation**. *IJCAI 2015*.

* Author: Qing-Yuan Jiang and Wu-Jun Li
* Contact: qyjiang24#gmail.com or liwujun#nju.edu.cn
* We use MNIST dataset as an example. If you want to run this demo on MNIST, please create MNIST.h5 by yourself. Specifically, MNIST.h5 contains four fields, i.e., "XDatabase", "XTest", "databaseL", "testL". XDatabase is the original features of training set and its dim is 60000X784. Similarly, XTest is 10000X784 features. databaseL and testL are one-hot labels with 60000X10 and 10000X10. Or, if you want to run it on your own dataset, please rewrite data-load function.
