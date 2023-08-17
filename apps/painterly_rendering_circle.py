"""
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_circles 2048 --max_radius 4.0
"""
import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import os
import time

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# pydiffvg.set_print_timing(True)
pydiffvg.set_print_timing(False)

gamma = 1.0

def main(args, logfile):
    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())

    start = time.time()
    
    #target = torch.from_numpy(skimage.io.imread('imgs/lena.png')).to(torch.float32) / 255.0
    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0

    print(f"===============image shape: {target.shape}================")
    if len(target.shape) == 2:
        target = torch.from_numpy(skimage.color.gray2rgb(skimage.io.imread(args.target))).to(torch.float32) / 255.0
    if target.shape[-1] == 4:
        # target = target[:,:,:3]
        target = torch.from_numpy(skimage.color.rgba2rgb(skimage.io.imread(args.target))).to(torch.float32) / 255.0
        # skimage.io.imsave(f'results/painterly_rendering/{args.exp_name}/target.png', skimage.color.rgba2rgb(skimage.io.imread(args.target)))
        
    print(f"===============image shape: {target.shape}================")

    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW
    #target = torch.nn.functional.interpolate(target, size = [256, 256], mode = 'area')
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    num_circles = args.num_circles
    max_radius = args.max_radius
    
    random.seed(1234)
    torch.manual_seed(1234)
    
    shapes = []
    shape_groups = []

    if args.use_random_position:
        for i in range(num_circles):
            radius = random.random()*max_radius
            center = [random.random()*canvas_width, random.random()*canvas_height]
            circle = pydiffvg.Circle(radius = torch.tensor(radius),
                                    center = torch.tensor(center))
            shapes.append(circle)
            circle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                                fill_color = torch.tensor([random.random(),
                                                                            random.random(),
                                                                            random.random(),
                                                                            random.random()]))
            shape_groups.append(circle_group)

    else:
        count_w = canvas_width//int(max_radius*2)
        count_h = canvas_height//int(max_radius*2)
        print(f"===============num_circles: {str(count_w*count_h)} ({count_w}*{count_h})================")
        logfile.write(f"num_circles: {str(count_w*count_h)} ({count_w}*{count_h})\n")
        logfile.write('\n')

        for i in range(count_w):
            for j in range(count_h):
                radius = random.random()*max_radius
                center = [max_radius+(max_radius*2)*i, max_radius+(max_radius*2)*j]
                circle = pydiffvg.Circle(radius = torch.tensor(radius),
                                        center = torch.tensor(center))
                shapes.append(circle)
                circle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                                    fill_color = torch.tensor([random.random(),
                                                                                random.random(),
                                                                                random.random(),
                                                                                random.random()]))
                shape_groups.append(circle_group)

    
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    # pydiffvg.imwrite(img.cpu(), 'results/painterly_rendering/init.png', gamma=gamma)
    pydiffvg.imwrite(img.cpu(), f'results/painterly_rendering/{args.exp_name}/init.png', gamma=gamma)
    # return

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for circle in shapes:
        circle.center.requires_grad = True
        points_vars.append(circle.center)
        circle.radius.requires_grad = True
        stroke_width_vars.append(circle.radius)
    for group in shape_groups:
        group.fill_color.requires_grad = True
        color_vars.append(group.fill_color)
    
    # Optimize
    point_lr = 2.0
    width_lr = 1.0
    color_lr = 0.01
    
    logfile.write(f'point_lr {point_lr}\n')
    logfile.write(f'width_lr {width_lr}\n')
    logfile.write(f'color_lr {color_lr}\n')

    # points_optim = torch.optim.Adam(points_vars, lr=1.0)
    points_optim = torch.optim.Adam(points_vars, lr=point_lr)
    if len(stroke_width_vars) > 0:
        # width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
        width_optim = torch.optim.Adam(stroke_width_vars, lr=width_lr)
    # color_optim = torch.optim.Adam(color_vars, lr=0.01)
    color_optim = torch.optim.Adam(color_vars, lr=color_lr)
    # Adam iterations.
    for t in range(args.num_iter):
        # print('iteration:', t)
        points_optim.zero_grad()
        if len(stroke_width_vars) > 0:
            width_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     t,   # seed
                     None,
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # Save the intermediate render.
        # pydiffvg.imwrite(img.cpu(), 'results/painterly_rendering/iter_{}.png'.format(t), gamma=gamma)
        # pydiffvg.imwrite(img.cpu(), 'results/painterly_rendering/' + args.exp_name + '/imgs/iter_{}.png'.format(t), gamma=gamma)
        if t == 100 or t == 200 or t == 500 or t == 1000 or t == 1500 or t == 2000:
            pydiffvg.imwrite(img.cpu(), 'results/painterly_rendering/' + args.exp_name + '/imgs/iter_{}.png'.format(t), gamma=gamma)
        if t == args.num_iter - 1:
            pydiffvg.imwrite(img.cpu(), f'results/painterly_rendering/{args.exp_name}/iter_{t}.png', gamma=gamma)
        
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        if args.use_lpips_loss:
            loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
        else:
            loss = (img - target).pow(2).mean()
        # print('render loss:', loss.item())
        if t == 0 or t % 50 == 0 or t == args.num_iter - 1:
            end = time.time() - start
            print(f'iteration: {t}, render loss:{loss.item()}, Processed time: {end:.2f}s')
            logfile.write(f'iteration: {t}, render loss:{loss.item()}, Processed time: {end:.2f}s\n')
    
        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        width_optim.step()
        color_optim.step()
        if args.max_radius_factor > 0:
            for circle in shapes:
                circle.radius.data.clamp_(1.0, max_radius*args.max_radius_factor)
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

        # if t % 10 == 0 or t == args.num_iter - 1:
        #     pydiffvg.save_svg('results/painterly_rendering/iter_{}.svg'.format(t),
        #                       canvas_width, canvas_height, shapes, shape_groups)
        pydiffvg.save_svg(f'results/painterly_rendering/{args.exp_name}/svgs/iter_{t}.svg',
                              canvas_width, canvas_height, shapes, shape_groups)
        if t == args.num_iter - 1:
            pydiffvg.save_svg(f'results/painterly_rendering/{args.exp_name}/iter_{t}.svg',
                              canvas_width, canvas_height, shapes, shape_groups)
    
    end = time.time() - start
    print('Processed time, time: %.5f s' % end)
    
    logfile.writelines("Process time: %.5f s" % end)
    # Render the final result.
    img = render(target.shape[1], # width
                 target.shape[0], # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    # Save the intermediate render.
    # pydiffvg.imwrite(img.cpu(), 'results/painterly_rendering/final.png'.format(t), gamma=gamma)
    # Convert the intermediate renderings to a video.
    # from subprocess import call
    # call(["ffmpeg", "-framerate", "24", "-i",
    #     f"results/painterly_rendering/{args.exp_name}/imgs/iter_%d.png", "-vb", "20M",
    #     f"results/painterly_rendering/{args.exp_name}/out.mp4"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    # parser.add_argument("--num_circles", type=int, default=512)
    parser.add_argument("--num_circles", type=int, default=100)
    parser.add_argument("--exp_name", type=str, default="test", help="experiment name, as the result folder name")
    parser.add_argument("--max_radius", type=float, default=50.0)
    # parser.add_argument("--max_radius", type=float, default=10.0)
    parser.add_argument("--max_radius_factor", type=float, default=2.0, help="put -1 if no limit")
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    # parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--num_iter", type=int, default=100)
    parser.add_argument("--use_random_position", dest='use_random_position', action='store_true')
    args = parser.parse_args()

    exist_ok = False
    if args.exp_name == 'test':
        print("Is test")
        exist_ok = True
    os.makedirs('results/painterly_rendering/' + args.exp_name, exist_ok=exist_ok)
    os.makedirs('results/painterly_rendering/' + args.exp_name + '/svgs', exist_ok=exist_ok)
    os.makedirs('results/painterly_rendering/' + args.exp_name + '/imgs', exist_ok=exist_ok)


    f = open("results/painterly_rendering/" + args.exp_name + "/report.txt", 'w')
    f.write(f"target img: {args.target}\n")
    f.write(f"use gpu: {str(torch.cuda.is_available())}\n")
    f.write(f"max_radius: {str(args.max_radius)}\n")
    if args.max_radius_factor > 0:
        f.write(f"max_radius_factor: {str(args.max_radius_factor)}\n")
    else:
        f.write(f"max_radius_factor: no limit ({str(args.max_radius_factor)})\n")
    f.write(f"use_lpips_loss: {str(args.use_lpips_loss)}\n")
    f.write(f"num_iter: {str(args.num_iter)}\n")
    f.write(f"use_circle: True\n")
    f.write(f"use_random_position: {str(args.use_random_position)}\n")
    if args.use_random_position:
        f.write(f"num_circles: {str(args.num_circles)}\n")
        f.write('\n')

    print(f"===============target img: {args.exp_name}================")

    
    main(args, f)
    
    f.close()
    