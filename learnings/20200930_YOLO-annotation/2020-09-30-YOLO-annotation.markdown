---
layout: post
title:  "YOLO Annotation"
date:   2019-09-30 21:37:00 +0530
categories: deep-learning
---

Lately, I have been [reading about YOLO][analyticsvdya-yolo] and this note is about how the annotations are stored for training YOLO.

Each image in the dataset should have a corresponding `txt` file. The bounding box in the image is represented by each line in the text file. The syntax of the line is as follows:

`class_id x y w h`

`x` and `y` are coordinates of the mid point of the bounding box. `w` and `h` are the width and height of the bounding box. The values for `x`, `y`, `w` and `h` are expressed relative to the image(ratio).

Let's say, `im_w` and `im_h` are the width and height of an image, and `(x_min, y_min)` and `(x_max, y_max)` are two diagonally opposite coordinates of a bounding box. To convert them into YOLO metrics, we find the midpoint of the bounding box- `((x_min + x_max) / 2 , (y_min + y_max) / 2)`. The width and height of bounding box is given by `x_max - x_min` and `y_max - y_min`. Then to express the values relative to image, we divide these values by `im_w` and `im_h`.

`x` = `(x_min + x_max) / (2 * im_w)`

`y` = `(y_min + y_max) / (2 * im_h)`

`w` = `(x_max - x_min) / im_w`

`h` = `(y_max - y_min) / im_h`

Below is the Python code.


{% highlight python %}
def convert(im_w, im_h, x_min, x_max, y_min, y_max):
    dw = 1./im_w
    dh = 1./im_h
    x = (x_min + x_max)/2.0
    y = (y_min + y_max)/2.0
    w = x_max - x_min
    h = y_max - y_min
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def deconvert(im_w, im_h, x, y, w, h):
    ox = float(x)
    oy = float(y)
    ow = float(w)
    oh = float(h)
    x = ox*im_w
    y = oy*im_h
    w = ow*im_w
    h = oh*im_h
    xmax = (((2*x)+w)/2)
    xmin = xmax-w
    ymax = (((2*y)+h)/2)
    ymin = ymax-h
    return [int(xmin),int(ymin),int(xmax),int(ymax)]
{% endhighlight %}


[Manivannan][manivannan] has shared an annotation [tool][tool] and wrote about [how to use it][annotation-tool]. Please note that his tool is in Python 2 and with some edits will work on Python 3.

[analyticsvdya-yolo]: https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-python/
[annotation-tool]: https://medium.com/@manivannan_data/yolo-annotation-tool-new-18c7847a2186
[tool]: https://github.com/ManivannanMurugavel/Yolo-Annotation-Tool-New-/
[manivannan]: https://github.com/ManivannanMurugavel