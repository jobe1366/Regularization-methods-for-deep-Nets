### :dart: _Design, train, test a network similar [LeNet](http://www.dengfanxin.cn/wp-content/uploads/2016/03/1998Lecun.pdf) and apply *Regularization* methods for sloveing Overfit problem  which has been uesed  [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)_



#### *step 1:* _train and test model without Regularization_
--- 


- *model summary*


![](protype%20model%20structure.PNG)



- *accuracy chart for train and test data*


![](acc-train-test-chart.PNG)



- *Confusion Matrix*
 

![](Confusion%20Matrix%20for%20protype%20model.PNG)



#### *step 2:* _train and test model along with Regularization_
---

- _Dropout layers:It is used to prevent over-fitting. In such a way that we randomly ignore a percentage of neurons during network training. The large number of parameters of the network and the strong dependence between neurons cause the power of each neuron to be limited and overfit on the most data._


- _Batch normalization:Instead of normalizing only the initial input, normalize the input of all layers.One of the advantages of using this method is that the weight of some neurons does not increase too much_


- _Augmentation:More training data for network training_


##### _Enhance model summary_
---


![](Enhanced%20model.PNG)


- _accuracy chart for train and test data_


![](acc-enhanced.PNG)


- _Confusion Matrix after regularization_


![](conf%20mtrx%20enhanced.PNG)



