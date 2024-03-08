import datetime
import time
from lxml.etree import XMLParser, parse, Element
from copy import deepcopy
import numpy as np
import cv2
import os.path as osp

class AverageMeter(object):

    def __init__(self, avg=None, count=1):
        self.reset()
        if avg is not None:
            self.val = avg
            self.avg = avg
            self.count = count
            self.sum = avg * count
    
    def __repr__(self) -> str:
        return f'{self.avg: .4f}'

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class time_printer:
    def __init__(self, name="process", enabled=True):
        self.name = name
        self.enabled = enabled
    
    def __enter__(self):
        if self.enabled:
            self.start = time.time()

    def __exit__(self, *args):
        if self.enabled:
            print(f'Time {self.name}: {time.time() - self.start:.6f}s')


def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return convert_sec_to_time(eta)


def convert_sec_to_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))


def create_vis_model_xml(in_file, out_file, num_actor=2, num_vis_capsules=0, num_vis_spheres=0, num_vis_planes=0):
    parser = XMLParser(remove_blank_text=True)
    tree = parse(in_file, parser=parser)
    geom_capsule = Element('geom', attrib={'fromto': '0 0 -10000 0 0 -9999', 'size': '0.02', 'type': 'capsule'})
    geom_sphere = Element('geom', attrib={'pos': '0 0 -10000', 'size': '0.02', 'type': 'sphere'})
    geom_plane = Element('geom', attrib={'pos': '0 0 -10000', 'size': '0.15 0.15 0.005', 'type': 'box'})

    root = tree.getroot().find('worldbody')
    body = root.find('body')
    for body_node in body.findall('.//body'):
        for joint_node in body_node.findall('joint')[1:]:
            body_node.remove(joint_node)
            body_node.insert(0, joint_node)

    for i in range(1, num_actor):
        new_body = deepcopy(body)
        #new_body.attrib['childclass'] = f'humanoid{i}'
        new_body.attrib['name'] = '%d_%s' % (i, new_body.attrib['name'])
        for node in new_body.findall(".//body"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        for node in new_body.findall(".//joint"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        for node in new_body.findall(".//freejoint"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        root.append(new_body)
    act_node = tree.find('actuator')
    act_node.getparent().remove(act_node)

    ind = 2
    for i in range(num_vis_capsules):
        root.insert(ind, deepcopy(geom_capsule))
        ind += 1
    for i in range(num_vis_spheres):
        root.insert(ind, deepcopy(geom_sphere))
        ind += 1
    for i in range(num_vis_planes):
        root.insert(ind, deepcopy(geom_plane))
        ind += 1
    
    ########## example to build the terrain ##########
    # https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-hfield
    # hfield = ""
    # if hfield is not None:
    #     asset = tree.getroot().find("asset")
    #     hfield_name = "/home/admin1/workspace/simulation/amp_imitator/field.png"
    #     # hfield_name = osp.join("./assets/mujoco_models/common", f"field.png")
    #     hfield_data = np.zeros((100, 100, 1))
    #     hfield_data[25:75, 25:75] = 15
    #     hfield_data[0, 0] = 256
    #     cv2.imwrite(hfield_name, hfield_data)

    #     asset.append(
    #                 Element(
    #                     "hfield",
    #                     {
    #                         "name": "floor",
    #                         "size": "4 4 2 0.2",
    #                         "file": hfield_name,
    #                     },
    #                 )
    #             )
    #     geom = tree.getroot().find("worldbody").find("geom")
    #     geom.attrib['hfield'] = 'floor'
    #     geom.attrib['type'] = 'hfield'
    #     geom.attrib['size'] = '16 16 1' ### not work for hfield but we need to set it
    #     geom.attrib['friction'] = '1. .1 .1'

    tree.write(out_file, pretty_print=True)



def create_terrain_vis_model_xml(in_file, out_file, terrain_info, num_actor=2, num_vis_capsules=0, num_vis_spheres=0, num_vis_planes=0):
    parser = XMLParser(remove_blank_text=True)
    tree = parse(in_file, parser=parser)
    geom_capsule = Element('geom', attrib={'fromto': '0 0 -10000 0 0 -9999', 'size': '0.02', 'type': 'capsule'})
    geom_sphere = Element('geom', attrib={'pos': '0 0 -10000', 'size': '0.02', 'type': 'sphere'})
    geom_plane = Element('geom', attrib={'pos': '0 0 -10000', 'size': '0.15 0.15 0.005', 'type': 'box'})

    root = tree.getroot().find('worldbody')
    body = root.find('body')
    for body_node in body.findall('.//body'):
        for joint_node in body_node.findall('joint')[1:]:
            body_node.remove(joint_node)
            body_node.insert(0, joint_node)

    for i in range(1, num_actor):
        new_body = deepcopy(body)
        #new_body.attrib['childclass'] = f'humanoid{i}'
        new_body.attrib['name'] = '%d_%s' % (i, new_body.attrib['name'])
        for node in new_body.findall(".//body"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        for node in new_body.findall(".//joint"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        for node in new_body.findall(".//freejoint"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        root.append(new_body)
    act_node = tree.find('actuator')
    act_node.getparent().remove(act_node)

    ind = 2
    for i in range(num_vis_capsules):
        root.insert(ind, deepcopy(geom_capsule))
        ind += 1
    for i in range(num_vis_spheres):
        root.insert(ind, deepcopy(geom_sphere))
        ind += 1
    for i in range(num_vis_planes):
        root.insert(ind, deepcopy(geom_plane))
        ind += 1
    
    ########## example to build the terrain ##########
    # https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-hfield
    hfield = ""
    if hfield is not None:
        asset = tree.getroot().find("asset")
        hfield_name = "/home/admin1/workspace/simulation/parserplus/field.png"
        # hfield_name = osp.join("./assets/mujoco_models/common", f"field.png")
        # hfield_data = np.zeros((100, 100, 1))
        # hfield_data[25:75, 25:75] = 15
        # hfield_data[0, 0] = 256
        hfield_data = terrain_info['hfield']
        cv2.imwrite(hfield_name, hfield_data)
        length, width, height_range, hfield_min = \
            terrain_info['length'], terrain_info['width'], terrain_info['hfield_range'], terrain_info['hfield_min']
        asset.append(
                    Element(
                        "hfield",
                        {
                            "name": "floor",
                            "size": "{} {} {} {}".format(length, width, height_range, np.abs(hfield_min)),
                            "file": hfield_name,
                        },
                    )
                )
        geom = tree.getroot().find("worldbody").find("geom")
        geom.attrib['hfield'] = 'floor'
        geom.attrib['type'] = 'hfield'
        geom.attrib['size'] = '16 16 1' ### not work for hfield but we need to set it
        geom.attrib['friction'] = '1. .1 .1'

    tree.find('size').attrib['njmax'] = "10000"
    tree.find('size').attrib['nconmax'] = "10000"
    tree.find('size').attrib['nstack'] = "10000000000000000000000"


    tree.write(out_file, pretty_print=True)