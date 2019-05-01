![](/uniofleeds.png)

# A Mathematical Model for Quorum Sensing in Pseudomonas aeruginosa Using Machine Learning

🎓This repository holds my final year and dissertation project during my time at the University of Leeds titled 'A Mathematical Model for Quorum Sensing in Pseudomonas aeruginosa Using Machine Learning' .

[PDF](https://github.com/jan-zajac/DISSERTATION_GIT/blob/master/DISSERATION.pdf)

## Abstract

*The harmful pathogen Pseudomonas aeruginosa has been researched for years, yet
even with the increased understanding of its behaviour, it continues to burden
healthcare services and infect vulnerable patients. A characteristic behaviour which
contributes to the pathogen’s virulence is known as quorum sensing. It allows the
bacteria to remain undetected by host immune systems, and only cause harm once
their colonies grow to overwhelmingly large populations. Quorum sensing is also
known to be affected by a multitude of internal and external factors. It was therefore
assumed that the variability of this behaviour could be captured by the modification of
coefficients in a mathematical model that was proved to be successful in describing
the quorum sensing behaviour. This project explored the use of machine learning
methods in the forecasting of this behaviour. Specifically, neural networks were used
in a multi-output regression problem to estimate the ordinary differential equation
coefficients of the mathematical model.*

*The project consisted of building technical foundation with the utilized software through
simpler examples. Artificial neural networks were successfully developed for estimating
the gradient of a generic linear equation, and the natural frequency and damping
ratio of a mass/spring/damper system – with the respective R2 values being 0.995
and 0.979. The insight gained was used to enhance the Pseudomonas aeruginosa
mathematical model.
The result was an artificial neural network capable of predicting
4 out of 11 coefficients ranging ± 40% from their original values stated in literature, with
a mean concentration error of 0.0637 units. The predictive capabilities of the enhanced
model give it life-saving potential in clinical applications through the possible
application of machine learning in diagnosis and treatment planning of infected patients.*

#### Dataset

The datasets used in this project were user-generated through the integration of ODEs (ordinary differential equations) (examples are available in downloadable directories). The project details 3 different systems:

1. A standard linear equation
2. A conventional engineering mass/spring/damper system
3. The mathematical model developed by Dockery (2001)

The corresponding targets for the supervised learning problem for the neural network are the ODE constants present in the different systems.

#### Requirements

* [Python v 3.6.1](https://www.python.org/)

#### Libraries Used

* [SciPy](https://www.scipy.org/)
* [Keras](https://keras.io/)
* [Talos](https://github.com/autonomio/talos)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

#### Usage

1. Download desired directory
2. Run DATA_GENERATION.py to generate two text files (input data and corresponding labels)
3. Run K-FOLD.py to carry out K-FOLD cross validation
4. Run NN_BUILD.py for prodcution of Keras neural network
5. Run RUN_MODEL.py to run model and evaluate

#### Citation

Zajac, J.Z., (2019). A Mathematical Model for Quorum Sensing in Pseudomonas aeruginosa Using Machine Learning

#### Notes

* Carried out in IDE (PyCharm CE)
* OS: macOS Mojave Version: 10.14.4

#### References

* Dockery, J. (2001). A Mathematical Model for Quorum Sensing in Pseudomonas
aeruginosa. Bulletin of Mathematical Biology, 63(1), pp.95-116.
