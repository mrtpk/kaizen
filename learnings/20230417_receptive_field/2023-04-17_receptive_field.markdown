---
layout: post
title:  "Receptive field of a convolution"
date:   2023-04-17 00:00:00 +0530
categories: deep learning
---


### Receptive field
The region in the input that is responsible for an output.![576314dfa67ab0b0b0c88e5eda04c285.png](:/8ca547f7aa1d4177b032c90a244bab84)

### Receptive field size
It is a recursive formula where we calculate receptive field from end to the start : `r(i−1) = s(i) × r(i) +(k(i) − s(i))`

An example:
![b1eb2fbed9189169fb296a5c03e1c521.png](:/758707ffcdc2466c8444ac0cf88ded2d)

### Effective Stride, S
`S(i-1) = S(i) * s(i)`
![9a6621aeeb60d93d0a5151064277faf5.png](:/f9a2858e6b4b4b03bb6d04af538e1521)

### Reference
+ https://theaisummer.com/receptive-field/
+ https://distill.pub/2019/computing-receptive-fields/
+ https://github.com/google-research/receptive_field/blob/master/RF_Keras_Applications.ipynb
