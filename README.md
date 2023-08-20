# Vector Graphics Non-photorealistic Rendering
向量圖形非寫實電腦計算繪製

本研究受到 [Differentiable Rasterizer for Vector Graphics](https://people.csail.mit.edu/tzumao/diffvg)的啟發，使用彩帶(向量不封閉曲線)、色塊(向量封閉曲線)以及圓形(向量圓形)等向量初始形狀來實作向量圖形的非寫實電腦計算繪製。

![teaser](https://github.com/hsiaohsc/vgnpr/assets/42900685/9db42de7-cd1e-4b3c-965e-5be7e3323ad8)


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

Painterly rendering
```
painterly_rendering.py [-h] [--num_paths NUM_PATHS]
                       [--max_width MAX_WIDTH] [--use_lpips_loss]
                       [--num_iter NUM_ITER] [--use_blob]
                       target
```
e.g.,
```
python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0 --use_lpips_loss
```

If you use diffvg in your academic work, please cite

```
@article{Li:2020:DVG,
    title = {Differentiable Vector Graphics Rasterization for Editing and Learning},
    author = {Li, Tzu-Mao and Luk\'{a}\v{c}, Michal and Gharbi Micha\"{e}l and Jonathan Ragan-Kelley},
    journal = {ACM Trans. Graph. (Proc. SIGGRAPH Asia)},
    volume = {39},
    number = {6},
    pages = {193:1--193:15},
    year = {2020}
}
```
