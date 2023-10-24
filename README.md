# Vector Graphics Non-photorealistic Rendering
向量圖形非寫實電腦計算繪製

[Project Page](https://hsiaohsc.github.io/thesis/)

This work is inspired by [Differentiable Rasterizer for Vector Graphics](https://people.csail.mit.edu/tzumao/diffvg), using vector-based initial shapes such as strips (unclosed vector paths), blobs (closed vector paths), and circles (vector circles) to create non-photorealistic vector graphics.

本研究受到 [Differentiable Rasterizer for Vector Graphics](https://people.csail.mit.edu/tzumao/diffvg) 的啟發，使用彩帶(向量不封閉曲線)、色塊(向量封閉曲線)以及圓形(向量圓形)等向量初始形狀來實作向量圖形的非寫實電腦計算繪製。

![teaser](https://github.com/hsiaohsc/vgnpr/assets/42900685/9db42de7-cd1e-4b3c-965e-5be7e3323ad8)


# Colab GUI Version
To use this code as a tool to generate non-photorealistic vector results, we provide a simple GUI user interface using Google Colab. \
[Run this code on Google Colab](https://colab.research.google.com/drive/1Hcilmt9R5BzWolY8kBaYIzj3joWbM3oH?usp=sharing)

# Install
```
git submodule update --init --recursive
conda install -y pytorch torchvision -c pytorch
conda install -y numpy
conda install -y scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
python setup.py install
```

# Building in debug mode

```
python setup.py build --debug install
```

# Run
```
cd apps
```

## **Vector Paths** 

including strips (unclosed vector paths) and  blobs (closed vector paths)
```
painterly_rendering_time.py [-h] [--exp_name EXP_NAME] [--num_paths NUM_PATHS]
                       [--max_width MAX_WIDTH] [--use_lpips_loss]
                       [--num_iter NUM_ITER] [--use_blob]
                       target
```
e.g.,
Strip
```
python painterly_rendering_time.py imgs/fuyue/01.jpg --exp_name test --num_paths 500 --max_width 5.0 --num_iter 200
```

Blob
```
python painterly_rendering_time.py imgs/fuyue/01.jpg --exp_name test --num_paths 500 --max_width 5.0 --num_iter 200 --use_blob
```

## **Vector Circle**
```
painterly_rendering_circle.py [-h] [--exp_name EXP_NAME] [--num_circles NUM_CIRCLES]
                       [--max_radius MAX_RADIUS] [--max_radius_factor MAX_RADIUS_FACTOR] [--use_lpips_loss]
                       [--num_iter NUM_ITER] [--use_random_position]
                       target
```
e.g.,
```
python painterly_rendering_circle.py imgs/fuyue/01.jpg --exp_name test --num_circles 500 --max_radius 5.0 --max_radius_factor 2.0 --num_iter 200 --use_random_position
```
or to generate circles in fixed position \
(number of circles will depend on the size of your input image and the `max_radius` parameter)
```
python painterly_rendering_circle.py imgs/fuyue/01.jpg --exp_name test --max_radius 5.0 --max_radius_factor 2.0 --num_iter 200
```
