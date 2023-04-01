---
layout: post
title:  "[draft] Convolution and correlation"
date:   2019-10-03 21:20:00 +0530
categories: deep-learning, DSP
mathjax: true
---

Convolution operation(denoted as $\ast$) between two signals, $f(t)$ and $g(t)$ is

$$
(f\ast g)(t) = \sum_a f(a) \cdot g(t-a)
$$

In convolution the signal $g(t)$ is flipped and convolved(passed through) on the input signal $f(t)$. Using convolution we look for what will be the output of the filter, $g(t)$ when it's input is $x(t)$.

Using correlation we check whether a signal $g(t)$ is present in a given noisy signal $x(t)$. One pratical application is template matching in computer vision. The correlation is given by

$$
(f\ast g)(t) = \sum_t f(t) \cdot g(t-a)
$$

TODO: Relationship between convolution and correlation using Numpy

TODO: Relationship between convolution, correlation and fft

**Reference:**
+ [A guide on convolution and correlation][correlation-dsp-guide]
+ [Understanding convolutions][understanding-convolution]
+ [Difference between convolution and correlation][dsp-exchange-difference-between-convolution-and-correlation]
+ [Correlation of two images using FFT][correlation-fft]

[dsp-exchange-difference-between-convolution-and-correlation]: https://dsp.stackexchange.com/questions/27451/the-difference-between-convolution-and-cross-correlation-from-a-signal-analysis/27453#comment122905_27453
[correlation-dsp-guide]: https://www.dspguide.com/ch7/3.htm
[understanding-convolution]: http://colah.github.io/posts/2014-07-Understanding-Convolutions/
[properties-convolution-and-correlation]: https://www.tutorialspoint.com/signals_and_systems/convolution_and_correlation.htm
[correlation-fft]: [https://stackoverflow.com/questions/58181398/how-to-find-correlation-between-two-images-using-numpy]