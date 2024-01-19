# Gussian Splatting 2D
A demo of 2d gaussian generating image.

## Usage

```cmd
python run.py
```

With 3000 sample points, you should have at least 16GB gpu memory.
The code only uses L1 loss for supervision. It will be better using other losses.

**Example of 3 random gaussian**
![Example](example.png)

**Result of 10 epoch**

![Result](result.png)

## Reference
[Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

[2D-Gaussian-Splatting](https://github.com/OutofAi/2D-Gaussian-Splatting)
