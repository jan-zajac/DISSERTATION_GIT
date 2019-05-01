# A Mathematical Model for Quorum Sensing in Pseudomonas aeruginosa Using Machine Learning

🎓This repository holds my final year and dissertation project during my time at the University of Leeds titled 'A Mathematical Model for Quorum Sensing in Pseudomonas aeruginosa Using Machine Learning' .

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

The datasets used in this project were user-generated through the integration of ODEs (ordinary differential equations). The project details 3 different systems.

<p align="center">
  <b>Some Links:</b><br>
  <a href = ![linearequation](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bdy%7D%7Bdt%7D%3Dm) </a>
  ![linearequation](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bdy%7D%7Bdt%7D%3Dm)
</p>

![linearequation](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bdy%7D%7Bdt%7D%3Dm)


#### Libraries Used

* [SciPy](https://www.scipy.org/)
* [Keras](https://keras.io/)
* [Talos](https://github.com/autonomio/talos)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

#### Notes

* Carried out in IDE (PyCharm CE)
* OS: macOS Mojave Version: 10.14.4
