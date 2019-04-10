# DEVOXX - Tools in Action - DL4J Demo

## Install

1. Compile and download dependencies with Maven : `mvn clean package`
2. Download data into a _data_ directory at the root of the project (see instructions [here](instructions.md))

## Webapp

To run the webapp, simply execute `mvn play2:run -pl webapp`

![alt text][webappScreenshot]

[webappScreenshot]: webapp/public/images/webapp.gif

## Train a model

Launch _TrainMainSolution_ from package `devoxx.dl4j.core.trainer`

## Use Tensorflow 2.0 to explore

As we wanted to experiment the interface betweend _ND4J_ and _Keras_, we built a Keras model with Tensorflow 2.0.

Everything is available [here](exploration/README.md)in the `exploration` directory, with all the explanations concerning the python virtualenv we used.

