# Quora Answer Classifier
Quora uses a combination of machine learning algorithms and moderation to ensure high-quality content on the site. High answer quality has helped Quora distinguish itself from other Q&A sites on the web.

In this work, we present a detailed study on the Quora Answer Classifier Challenge. As real datasets tend to be messy and pre-processing is often much more important than any specific modeling decisions, a great part of our work would be a study and analysis of the provided data in order to increase the classifiers' accuracy.

Further, we not only propose a solution to the classification problem, but also use the provided dataset in order to compare different classifiers (Naive Bayes, Logistic Regression, Neural Network and AdaBoost). We demonstrate the benefit of pre-processing methods such as normalization, Forward Backward Greedy Feature Selection, Random Forest Feature Selection and Lasso Feature Selection by achieving a classification accuracy of 81.4% using a simple Logistic Regression classifier, third greatest score in the contest. More information about the challenge can be obtained [here](https://www.quora.com/challenges#answer_classifier).

## Developers
 - Amanjot Kaur
 - Guilherme Trein

## Running the simulation
The simulation script is straightforward. The `quora.py` is the entry point of the simulation and contains several experiments that were studied in order to propose a simple solution to the Quora Answer Classifier. To run the simulation, issue the command:

    $ python quora.py

Note that the simulation requires the libraries [scikit-learn](http://scikit-learn.org/stable/) and [numpy](http://www.numpy.org/). It is recommended to install them using [virtualenv](http://www.virtualenv.org/en/latest/).


## Acknowledgements
We are grateful to Kevin Swersky for guiding us through the problem, and many helpful discussions and suggestions. We also thank Navdeep Jaitly for the provided guidelines on Neural Networks implementation. Finally, we thank Professor Dr. Richard Zemel for teaching us many of the concepts involved in this project.

 - [Richard Zemel](http://www.cs.toronto.edu/~zemel), Professor Dept. of Computer Science at University of Toronto, CSC2515 (Fall 2013)
 - [Kevin Swersky](http://www.cs.toronto.edu/~kswersky/), Ph.D. Student at University of Toronto and TA of CSC2515 (Fall 2013).
 - [Navdeep Jaitly](http://www.cs.toronto.edu/~ndjaitly/), Ph.D. Student at University of Toronto and TA of CSC2515 (Fall 2013).

