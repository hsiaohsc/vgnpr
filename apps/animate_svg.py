import pydiffvg
import argparse

import torch
import xml.etree.ElementTree as etree
from xml.dom import minidom
import os

def save_animated_svg(folder, iters, filename, step=1, dur=0.1):
    if not os.path.exists(folder):
        print("[Error] Folder doesn't exist!")
        return

    # Get width and height
    fname = os.path.join(folder, f'iter_0.svg')
    if not os.path.exists(fname):
        print(f"[Error] SVG file {fname} not found...")
        return
    pictree = etree.parse(fname)
    picroot = pictree.getroot()
    if 'width' in picroot.attrib:
        width = pydiffvg.parse_int(picroot.attrib['width'])
    else:
        print('Warning: Can\'t find canvas width.')
        width = 800
    if 'height' in picroot.attrib:
        height = pydiffvg.parse_int(picroot.attrib['height'])
    else:
        print('Warning: Can\'t find canvas height.')
        height = 538

    # New doc
    root = etree.Element('svg')
    root.set('version', '1.1')
    root.set('xmlns', 'http://www.w3.org/2000/svg')
    root.set('width', str(width))
    root.set('height', str(height))
    defs = etree.SubElement(root, 'defs')
    # g = etree.SubElement(root, 'g')
    # g.set('opacity', '0')

    animation_id = 0
    for i in range(0, iters, step):
        fname = os.path.join(folder, f'iter_{i}.svg')
        if not os.path.exists(fname):
            print(f"[Warning] SVG file {fname} not found...")
            continue
        pictree = etree.parse(fname)
        picroot = pictree.getroot()
        for child in picroot:
            tag = pydiffvg.remove_namespaces(child.tag)
            if tag == 'g':
                child.tag = 'g'
                child.set('opacity', '0')
                # child.set('visibility', 'hidden')

                # Clear path tags
                for path in child:
                    if pydiffvg.remove_namespaces(path.tag) == 'path':
                        path.tag = 'path'

                # Appear animation
                ani_appear = etree.SubElement(child, 'animate')
                ani_appear.set('id', str(animation_id))
                ani_appear.set('attributeType', 'CSS')
                # ani_appear.set('attributeName', 'visibility')
                # ani_appear.set('from', 'hidden')
                # ani_appear.set('to', 'visible')
                ani_appear.set('attributeName', 'opacity')
                ani_appear.set('from', '0')
                ani_appear.set('to', '1')
                ani_appear.set('dur', f'{dur}s')
                ani_appear.set('fill', 'freeze')
                if i == 0:
                    ani_appear.set('begin', f'0;{iters*2-1}.end')
                else:
                    ani_appear.set('begin', f'{animation_id-2}.end-{dur/2}s')
                    # ani_appear.set('begin', f'{(animation_id-1)*dur-(dur/2)}')
                animation_id+=1
                
                # Disappear animation
                ani_disappear = etree.SubElement(child, 'animate')
                ani_disappear.set('id', str(animation_id))
                ani_disappear.set('attributeType', 'CSS')
                # ani_disappear.set('attributeName', 'visibility')
                # ani_disappear.set('from', 'visible')
                # ani_disappear.set('to', 'hidden')
                ani_disappear.set('attributeName', 'opacity')
                ani_disappear.set('from', '1')
                ani_disappear.set('to', '0')
                ani_disappear.set('dur', f'{dur}s')
                # ani_disappear.set('dur', '1s')
                ani_disappear.set('fill', 'freeze')
                ani_disappear.set('begin', f'{animation_id-1}.end')
                animation_id+=1

                root.append(child)

    with open(filename, "w") as f:
        f.write(pydiffvg.prettify(root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="svg folder")
    parser.add_argument("iters", type=int, default=299, help="number of svg iterations included for animation")
    parser.add_argument("filename", type=str, help="store svg filepath")
    parser.add_argument("--step", type=int, default=1, help="step for iteration")
    parser.add_argument("--time", type=int, default=20, help="animation length")
    args = parser.parse_args()
    
    
    # duration = args.time / ((args.iters+1) / args.step)
    # print(duration)
    save_animated_svg(args.folder, args.iters, args.filename, args.step, 0.1)

    print("Animation saved...")