# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import sys
import random
import argparse
import json
import os
from datetime import datetime as dt
import numpy as np
import errno
# from generate.movement_record import MovementRecord
import itertools
import logging
sys.path.insert(0, os.getcwd())



"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
    import bpy
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import generate.utils as utils
        import generate.actions as actions
    except ImportError as e:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import "
              "utils.py. You may need to add a .pth file to the site-packages "
              "of Blender's bundled python with a command like this:\n "
              "echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth"  # noQA
              "\nWhere $BLENDER is the directory where Blender is installed, "
              "and $VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument(
    '--base_scene_blendfile',
    # default='data/base_scene.blend',
    default='data/base_scene_withAxes.blend',
    help="Base blender file on which all scenes are based; includes " +
         "ground plane, lights, and camera.")
parser.add_argument(
    '--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the 'materials' and 'shapes' fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument(
    '--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument(
    '--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument(
    '--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument(
    '--min_objects', default=5, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument(
    '--max_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument(
    '--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument(
    '--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart; making resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument(
    '--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument(
    '--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument(
    '--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. " +
         "Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument(
    '--num_images', default=1, type=int,
    help="The number of images to render")
parser.add_argument(
    '--parallel_mode', action='store_true',
    help="Set if running on multiple nodes/GPUs. Will use lock files "
         "to synchronize.")
parser.add_argument(
    '--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to rendered images and JSON scenes")
parser.add_argument(
    '--split', default='new',
    help="Name of the split for which we are rendering. " +
         "This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument(
    '--output_dir', default="./",
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
# parser.add_argument(
#     '--output_image_dir', default='../output/images/',
#     help="The directory where output images will be stored. It will be " +
#          "created if it does not exist.")
# parser.add_argument(
#     '--output_scene_dir', default='../output/scenes/',
#     help="The directory where output JSON scene structures will be stored. " +
#          "It will be created if it does not exist.")
parser.add_argument(
    '--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
# parser.add_argument(
#     '--output_blend_dir', default='output/blendfiles',
#     help="The directory where blender scene files will be stored, if the " +
#          "user requested that these files be saved using the " +
#          "--save_blendfiles flag; in this case it'll be created if it does " +
#          "not already exist.")
parser.add_argument(
    '--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument(
    '--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument(
    '--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument(
    '--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument(
    '--cpu', action='store_true', default=False,
    help="Setting true disables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "GPU rendering to work. For specifying a GPU, use "
         "CUDA_VISIBLE_DEVICES before running singularity.")
parser.add_argument(
    '--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument(
    '--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument(
    '--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument(
    '--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument(
    '--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument(
    '--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument(
    '--render_num_samples', default=128, type=int,  # CLEVR was 512
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument(
    '--render_min_bounces', default=2, type=int,  # default 8
    help="The minimum number of bounces to use for rendering.")
parser.add_argument(
    '--render_max_bounces', default=2, type=int,  # default 8
    help="The maximum number of bounces to use for rendering.")
parser.add_argument(
    '--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")

# Video options
parser.add_argument(
    '--num_frames', default=300, type=int,
    help="Number of frames to render.")
parser.add_argument(
    '--num_flips', default=10, type=int,
    help="Number of flips to render.")
parser.add_argument(
    '--fps', default=24, type=int,
    help="Video FPS.")
parser.add_argument(
    '--render', default=True, type=bool,
    help="Render the video. Otherwise will only store the blend file.")
parser.add_argument(
    "--random_camera", help="Render the video with random camera motion",
    action="store_true")
parser.add_argument(
    "--max_motions",
    help="Number of max objects to move in the single object case. "
         "This ensures the actions are sparser, and random perf lower.",
    type=int, default=999999)

parser.add_argument(
    '-d', '--debug', action='store_true',
    help="Run in debug mode. Will crash on exceptions.")
parser.add_argument(
    '--suppress_blender_logs', action='store_true',
    help="Dont print extra blender logs.")
parser.add_argument(
    "-v", "--verbose", help="increase output verbosity",
    action="store_true")

random.seed(42)
np.random.seed(42)


def mkdir_p(path):
    """
    Make all directories in `path`. Ignore errors if a directory exists.
    Equivalent to `mkdir -p` in the command line, hence the name.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def lock(fpath):
    lock_fpath = fpath + '.lock'
    if os.path.exists(fpath) or os.path.exists(lock_fpath):
        return False
    try:
        mkdir_p(lock_fpath)
        return True
    except Exception as e:
        logging.warning('Unable to lock {} due to {}'.format(fpath, e))
        return False


def unlock(fpath):
    lock_fpath = fpath + '.lock'
    try:
        os.rmdir(lock_fpath)
    except Exception as e:
        logging.warning('Maybe some other job already finished {}. Got {}'
                        .format(fpath, e))


def main(args):
    num_digits = 6
    prefix = '%s_%s_' % (args.filename_prefix, args.split)
    img_template = '%s%%0%dd.avi' % (prefix, num_digits)
    scene_template = '%s%%0%dd.json' % (prefix, num_digits)
    blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
    args.output_image_dir = os.path.join(args.output_dir, 'images')
    args.output_scene_dir = os.path.join(args.output_dir, 'scenes')
    args.output_blend_dir = os.path.join(args.output_dir, 'blend')
    img_template = os.path.join(args.output_image_dir, img_template)
    scene_template = os.path.join(args.output_scene_dir, scene_template)
    blend_template = os.path.join(args.output_blend_dir, blend_template)

    mkdir_p(args.output_image_dir)
    mkdir_p(args.output_scene_dir)
    if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
        mkdir_p(args.output_blend_dir)

    all_scene_paths = []
    for i in range(args.num_images):
        img_path = img_template % (i + args.start_idx)
        if not lock(img_path):
            continue
        logging.info('Working on {}'.format(img_path))
        scene_path = scene_template % (i + args.start_idx)
        all_scene_paths.append(scene_path)
        blend_path = None
        if args.save_blendfiles == 1:
            blend_path = blend_template % (i + args.start_idx)
        num_objects = random.randint(args.min_objects, args.max_objects)
        try:
            render_scene(
                args,
                num_objects=num_objects,
                output_index=(i + args.start_idx),
                output_split=args.split,
                output_image=img_path,
                output_scene=scene_path,
                output_blendfile=blend_path,
            )
        except Exception as e:
            if args.debug:
                unlock(img_path)
                raise e
            logging.warning('Didnt work for {} due to {}. Ignoring for now..'
                            .format(img_path, e))
        unlock(img_path)
        logging.info('Done for {}'.format(img_path))

    # After rendering all images, combine the JSON files for each scene into a
    # single JSON file.
    all_scenes = []
    for scene_path in all_scene_paths:
        with open(scene_path, 'r') as f:
            all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': args.date,
            'version': args.version,
            'split': args.split,
            'license': args.license,
        },
        'scenes': all_scenes
    }
    print("#" * 15,'\n')
    print(args.output_scene_file, "\n")
    with open(args.output_scene_file, 'w') as f:
        json.dump(output, f)


def rand(L):
    return 2.0 * L * (random.random() - 0.5)


def setup_scene(
    args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
  ):

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'split': output_split,
        'image_index': output_index,
        'image_filename': os.path.basename(output_image),
        'objects': [],
        'directions': {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    # Add random jitter to camera position
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)
    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects['Camera']

    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(
                args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(
                args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(
                args.fill_light_jitter)

    # objects = cup_game(scene_struct, num_objects, args, camera)
    objects, blender_objects = add_random_objects(
        scene_struct, num_objects, args, camera)
    record = MovementRecord(blender_objects, args.num_frames)
    actions.random_objects_movements(
        objects, blender_objects, args, args.num_frames, args.min_dist,
        record, max_motions=args.max_motions)

    # Render the scene and dump the scene data structure
    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    scene_struct['movements'] = record.get_dict()
    # TODO need to add BB saving
    obj_names = ["_".join(obj[att] for att in ["size", "color", "shape", "material", "instance"]) for obj in objects]
    objects_bb = {obj:[] for obj in obj_names}

    for frame in range(bpy.data.scenes[0].frame_start, bpy.data.scenes[0].frame_end):
        bpy.data.scenes[0].frame_set(frame)
        for obj in objects:
            bb = camera_view_bounds_2d(bpy.context.scene,
                                       bpy.context.scene.camera,
                                       bpy.data.objects[obj["instance"]])
            obj_name = "_".join(obj[att] for att in ["size", "color", "shape", "material", "instance"])
            objects_bb[obj_name].append(bb)

    scene_struct["camera"] = np.array(bpy.data.objects['Camera'].location).tolist()

    base_path = os.path.split(output_scene)[0]
    bb_name = os.path.split(output_scene)[-1].split('.')[0]

    # In case we want to generate object masks:
    # render_shadeless(blender_objects, path=output_image[:-4] + '_mask.png')

    with open(os.path.join(base_path,bb_name+"_bb.json"),'w') as f:
        json.dump(objects_bb, f, indent=2)

    with open(output_scene, 'w') as f:
        json.dump(scene_struct, f, indent=2)


def camera_as_planes(scene, obj):
    """
    Return planes in world-space which represent the camera view bounds.
    """
    from mathutils.geometry import normal

    camera = obj.data
    # normalize to ignore camera scale
    matrix = obj.matrix_world.normalized()
    frame = [matrix * Vector(v) for v in camera.view_frame(scene)]
    origin = matrix.to_translation()

    planes = []
    is_persp = (camera.type != 'ORTHO')
    for i in range(4):
        # find the 3rd point to define the planes direction
        if is_persp:
            frame_other = origin
        else:
            frame_other = frame[i] + matrix.col[2].xyz

        n = normal(frame_other, frame[i - 1], frame[i])
        d = -n.dot(frame_other)
        planes.append((n, d))

    if not is_persp:
        # add a 5th plane to ignore objects behind the view
        n = normal(frame[0], frame[1], frame[2])
        d = -n.dot(origin)
        planes.append((n, d))

    return planes


def side_of_plane(p, v):
    return p[0].dot(v) + p[1]


def is_segment_in_planes(p1, p2, planes):
    dp = p2 - p1

    p1_fac = 0.0
    p2_fac = 1.0

    for p in planes:
        div = dp.dot(p[0])
        if div != 0.0:
            t = -side_of_plane(p, p1)
            if div > 0.0:
                # clip p1 lower bounds
                if t >= div:
                    return False
                if t > 0.0:
                    fac = (t / div)
                    p1_fac = max(fac, p1_fac)
                    if p1_fac > p2_fac:
                        return False
            elif div < 0.0:
                # clip p2 upper bounds
                if t > 0.0:
                    return False
                if t > div:
                    fac = (t / div)
                    p2_fac = min(fac, p2_fac)
                    if p1_fac > p2_fac:
                        return False

    ## If we want the points
    # p1_clip = p1.lerp(p2, p1_fac)
    # p2_clip = p1.lerp(p2, p2_fac)
    return True


def point_in_object(obj, pt):
    xs = [v[0] for v in obj.bound_box]
    ys = [v[1] for v in obj.bound_box]
    zs = [v[2] for v in obj.bound_box]
    pt = obj.matrix_world.inverted() * pt
    return (min(xs) <= pt.x <= max(xs) and
            min(ys) <= pt.y <= max(ys) and
            min(zs) <= pt.z <= max(zs))


def object_in_planes(obj, planes):
    from mathutils import Vector

    matrix = obj.matrix_world
    box = [matrix * Vector(v) for v in obj.bound_box]
    for v in box:
        if all(side_of_plane(p, v) > 0.0 for p in planes):
            # one point was in all planes
            return True

    # possible one of our edges intersects
    edges = ((0, 1), (0, 3), (0, 4), (1, 2),
             (1, 5), (2, 3), (2, 6), (3, 7),
             (4, 5), (4, 7), (5, 6), (6, 7))
    if any(is_segment_in_planes(box[e[0]], box[e[1]], planes)
           for e in edges):
        return True


    return False


def objects_in_planes(objects, planes, origin):
    """
    Return all objects which are inside (even partially) all planes.
    """
    return [obj for obj in objects
            if point_in_object(obj, origin) or
               object_in_planes(obj, planes)]

def select_objects_in_camera():
    from bpy import context
    scene = context.scene
    origin = scene.camera.matrix_world.to_translation()
    planes = camera_as_planes(scene, scene.camera)
    objects_in_view = objects_in_planes(scene.objects, planes, origin)
    return objects_in_view

class Box:

    dim_x = 1
    dim_y = 1

    def __init__(self, min_x, min_y, max_x, max_y, dim_x=dim_x, dim_y=dim_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.dim_x = dim_x
        self.dim_y = dim_y

    @property
    def x(self):
        return round(self.min_x * self.dim_x)

    @property
    def y(self):
        return round(self.dim_y - self.max_y * self.dim_y)

    @property
    def width(self):
        return round((self.max_x - self.min_x) * self.dim_x)

    @property
    def height(self):
        return round((self.max_y - self.min_y) * self.dim_y)

    def __str__(self):
        return "<Box, x=%i, y=%i, width=%i, height=%i>" % \
               (self.x, self.y, self.width, self.height)

    def to_tuple(self):
        if self.width == 0 or self.height == 0:
            return (0, 0, 0, 0)
        return (self.x, self.y, self.width, self.height)


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    me = me_ob.to_mesh(scene, True, "PREVIEW")
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            #if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac
    box = Box(min_x, min_y, max_x, max_y, dim_x, dim_y)
    return [box.x, box.y, box.width, box.height]


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def render_scene(
        args,
        num_objects=5,
        output_index=0,
        output_split='none',
        output_image='render.png',
        output_scene='render_json',
        output_blendfile=None):
    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    bpy.ops.screen.frame_jump(end=False)
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    render_args.image_settings.file_format = 'AVI_JPEG'
    # Video params
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = args.num_frames  # same as kinetics
    render_args.fps = args.fps

    if args.cpu is False:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        else:
            cycles_prefs = bpy.context.user_preferences.addons[
                'cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'
            # # In case more than 1 device passed in, use only the first one
            # Not effective, CUDA_VISIBLE_DEVICES before running singularity
            # works fastest.
            # if len(cycles_prefs.devices) > 2:
            #     for device in cycles_prefs.devices:
            #         device.use = False
            #     cycles_prefs.devices[1].use = True
            #     print('Too many GPUs ({}). Using {}. Set only 1 before '
            #           'running singularity.'.format(
            #               len(cycles_prefs.devices),
            #               cycles_prefs.devices[1]))

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.cpu is False:
        bpy.context.scene.cycles.device = 'GPU'

    if output_blendfile is not None and os.path.exists(output_blendfile):
        logging.info('Loading pre-defined BLEND file from {}'.format(
            output_blendfile))
        bpy.ops.wm.open_mainfile(filepath=output_blendfile)
    else:
        setup_scene(
            args, num_objects, output_index, output_split,
            output_image, output_scene)
    print_camera_matrix()
    if args.random_camera:
        add_random_camera_motion(args.num_frames)
    if output_blendfile is not None and not os.path.exists(output_blendfile):
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    max_num_render_trials = 10
    if args.render:
        while max_num_render_trials > 0:
            try:
                if args.suppress_blender_logs:
                    # redirect output to log file
                    logfile = '/dev/null'
                    open(logfile, 'a').close()
                    old = os.dup(1)
                    sys.stdout.flush()
                    os.close(1)
                    os.open(logfile, os.O_WRONLY)
                bpy.ops.render.render(animation=True)
                if args.suppress_blender_logs:
                    # disable output redirection
                    os.close(1)
                    os.dup(old)
                    os.close(old)
                break
            except Exception as e:
                max_num_render_trials -= 1
                print(e)


def print_camera_matrix():
    # from
    # https://blender.stackexchange.com/questions/16472/how-can-i-get-the-cameras-projection-matrix
    camera = bpy.data.objects['Camera']
    render = bpy.context.scene.render
    modelview_matrix = camera.matrix_world.inverted()
    projection_matrix = camera.calc_matrix_camera(
        render.resolution_x,
        render.resolution_y,
        render.pixel_aspect_x,
        render.pixel_aspect_y,
    )
    final_mat = projection_matrix * modelview_matrix
    print('Overall camera matrix:', final_mat)


def get_new_camera_location():
    # Don't move in X and Y at the same time, as it crosses the 0,0,z point
    # which is a singularity
    new_x, new_y, new_z = None, None, None
    if np.random.random() > 0.5:
        # Move in X
        new_x = np.random.choice([-10, 10])
    else:
        # Move in Y
        new_y = np.random.choice([-10, 10])
    new_z = np.random.choice([8, 10, 12])
    return new_x, new_y, new_z


def add_random_camera_motion(num_frames):
    # Now go through these locations in a random order
    shift_interval = 30
    # Start from the same position everytime, as I want to be able to track
    # positions
    add_camera_position(0, (None, None, None))
    for frame_id in range(shift_interval, num_frames, shift_interval):
        last_loc = get_new_camera_location()
        add_camera_position(frame_id, last_loc)
    add_camera_position(num_frames, last_loc)


def add_camera_position(frame_id, loc):
    obj = bpy.data.objects['Camera']
    if loc[0] is not None:
        obj.location.x = loc[0]
    if loc[1] is not None:
        obj.location.y = loc[1]
    if loc[2] is not None:
        obj.location.z = loc[2]
    obj.keyframe_insert(data_path='location', frame=frame_id)


def add_random_objects(scene_struct, num_objects, args, camera):
    """
    Add random objects to the current blender scene
    """

    # Load the property file
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = list(properties['sizes'].items())

    shape_color_combos = None
    if args.shape_color_combos_json is not None:
        with open(args.shape_color_combos_json, 'r') as f:
            shape_color_combos = list(json.load(f).items())

    positions = []
    objects = []
    blender_objects = []
    for i in range(num_objects):
        if i == 0:
            # first element is the small shiny gold "snitch"!
            size_name, r = "small", 0.3  # slightly larger than small
            obj_name, obj_name_out = 'Spl', 'spl'
            color_name = "gold"
            rgba = [1.0, 0.843, 0.0, 1.0]
            mat_name, mat_name_out = "MyMetal", "metal"
        elif i == 1:
            # second element is a medium cone
            size_name, r = "medium", 0.5
            obj_name, obj_name_out = 'Cone', 'cone'
            color_name, rgba = random.choice(list(
                color_name_to_rgba.items()))
            mat_name, mat_name_out = random.choice(material_mapping)
        elif i == 2:
            # third element is a large cone
            size_name, r = "large", 0.75
            obj_name, obj_name_out = 'Cone', 'cone'
            color_name, rgba = random.choice(list(
                color_name_to_rgba.items()))
            mat_name, mat_name_out = random.choice(material_mapping)
        else:
            # Choose a random size
            size_name, r = random.choice(size_mapping)
            # Choose random color and shape
            if shape_color_combos is None:
                obj_name, obj_name_out = random.choice(object_mapping)
                color_name, rgba = random.choice(list(
                    color_name_to_rgba.items()))
            else:
                obj_name_out, color_choices = random.choice(shape_color_combos)
                color_name = random.choice(color_choices)
                obj_name = [k for k, v in object_mapping
                            if v == obj_name_out][0]
                rgba = color_name_to_rgba[color_name]
            # Choose a random material
            mat_name, mat_name_out = random.choice(material_mapping)

        # Try to place the object, ensuring that we don't intersect any
        # existing objects and that we are more than the desired margin away
        # from all existing objects along all cardinal directions.
        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then
            # delete all the objects in the scene and start over.
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_random_objects(scene_struct, num_objects, args,
                                          camera)
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            # Check to make sure the new object is further than min_dist from
            # all other objects, and further than margin along the four
            # cardinal directions
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < args.min_dist:
                    dists_good = False
                    break
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = scene_struct['directions'][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < args.margin:
                        logging.debug('{} {} {}'.format(
                            margin, args.margin, direction_name))
                        logging.debug('BROKEN MARGIN!')
                        margins_good = False
                        break
                if not margins_good:
                    break
            if dists_good and margins_good:
                break

        # For cube, adjust the size a bit
        if obj_name == 'Cube':
            r /= math.sqrt(2)

        # Choose random orientation for the object.
        theta = 360.0 * random.random()

        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        # Actually add material
        utils.add_material(mat_name, Color=rgba)

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            'shape': obj_name_out,
            'size': size_name,
            'sized': r,
            'material': mat_name_out,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
            'instance': obj.name,
        })
    return objects, blender_objects


def cup_game(scene_struct, num_objects, args, camera):
    # make some random objects
    # objects, blender_objects = add_random_objects(
    objects, blender_objects = add_cups(
        scene_struct, num_objects, args, camera)
    bpy.ops.screen.frame_jump(end=False)
    # from https://blender.stackexchange.com/a/70478
    add_flips(blender_objects, num_flips=args.num_flips,
              total_frames=args.num_frames)
    animate_camera(args.num_frames)
    return objects


def animate_camera(num_frames):
    path = [
        (0, -10, 10),
        (-10, 0, 10),
        (0, 10, 10),
        (10, 0, 5),
    ]
    shift_interval = 20
    cur_pos_id = -1
    obj = bpy.data.objects['Camera']
    for frame_id in range(0, num_frames, shift_interval):
        obj.keyframe_insert(data_path='location', frame=frame_id)
        cur_pos_id = (cur_pos_id + 1) % len(path)
        obj.location.x = path[cur_pos_id][0]
        obj.location.y = path[cur_pos_id][1]
        obj.location.z = path[cur_pos_id][2]


def add_cups(scene_struct, num_objects, args, camera):
    """
    Add random objects to the current blender scene
    """
    # Load the property file
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = list(properties['sizes'].items())

    # shape_color_combos = None
    # if args.shape_color_combos_json is not None:
    #     with open(args.shape_color_combos_json, 'r') as f:
    #         shape_color_combos = list(json.load(f).items())

    positions = []
    objects = []
    blender_objects = []

    # Choose a random size, same for all cups
    size_name, r = size_mapping[0]
    first_cup_x = 0
    first_cup_y = 0

    # obj_name, obj_name_out = random.choice(object_mapping)
    obj_name, obj_name_out = [el for el in object_mapping
                              if el[1] == 'cylinder'][0]
    color_name, rgba = random.choice(list(color_name_to_rgba.items()))

    # If using combos
    # obj_name_out, color_choices = random.choice(shape_color_combos)
    # color_name = random.choice(color_choices)
    # obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
    # rgba = color_name_to_rgba[color_name]

    # Attach a random material
    mat_name, mat_name_out = random.choice(material_mapping)

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
        r /= math.sqrt(2)

    # Choose random orientation for the object.
    # theta = 360.0 * random.random()
    theta = 0.0

    for i in range(num_objects):
        x = first_cup_x + i * 1.5
        y = first_cup_y

        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        utils.add_material(mat_name, Color=rgba)

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            'shape': obj_name_out,
            'size': size_name,
            'material': mat_name_out,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color_name,
        })
    return objects, blender_objects


def add_flips(blender_objects, num_flips=10, total_frames=300):
    # add current locations as a keyframe
    current_frame = 0
    bpy.context.scene.frame_set(current_frame)
    for obj in blender_objects:
        obj.keyframe_insert(data_path='location')

    frames_per_flip = total_frames // num_flips
    for flip_id in range(num_flips):
        # select random 2 cups to flip
        end_frame = min(current_frame + frames_per_flip - 1, total_frames)
        actions.add_flip(
            blender_objects, start_frame=current_frame, end_frame=end_frame)
        current_frame = end_frame + 1
    bpy.ops.screen.frame_jump(end=False)


def compute_all_relationships(scene_struct, eps=0.2):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i]
    then object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below':
            continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj1 == obj2:
                    continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
                all_relationships[name].append(sorted(list(related)))
    return all_relationships


class MovementRecord:
    def __init__(self, objects, total_frames):
        """
        Args:
            objects (list of blender objects)
        """
        # initialize a list for each object
        self.total_frames = total_frames
        self.timeline = {}
        self.contains = {}
        for obj in objects:
            self.timeline[obj] = []
            # store for each frame, what is contained
            self.contains[obj] = list(itertools.repeat(None, total_frames + 1))

    def insert(self, obj, action, other_obj, start_frame, end_frame):
        """
        Args:
            obj: The blender object that was acted upon
            action: The function/action op that was executed
            other_obj: Other blender obj involved in this.
                Only useful for contains op
            start_frame: The start_frame of the action
            end_frame: The end_frame of the action
        """
        self.timeline[obj].append((
            action,
            other_obj,
            start_frame, end_frame))
        logging.debug('Recorded action for {}: {}'.format(
            obj, self.human_readable_interval(self.timeline[obj][-1])))
        # If contains, also update the contains data structure
        if action.__name__ == '_contain':
            # Assume it will be contained for ever, until the same object hits
            # a pick_place
            assert obj != other_obj, '{} can not contain itself!'.format(
                obj)  # will lead to infinite recursion when checking
            for frame_id in range(start_frame, self.total_frames + 1):
                # Nothing should already be contained
                assert self.contains[obj][frame_id] is None, \
                    '{} already contains {} at frame {}. ' \
                    'Cant contain {} (btw {} and {}) now also..?' \
                    'This may be because I use generous timing for contains ' \
                    'op, i.e. since the the cone picks up, it is counted as ' \
                    'contains. Anyway, ignore this setup.'.format(
                        obj, self.contains[obj][frame_id], frame_id,
                        other_obj, start_frame, end_frame)
                self.contains[obj][frame_id] = other_obj
            logging.debug('{} contains {} as of {}'.format(
                obj, other_obj, start_frame))
        elif action.__name__ == '_pick_place':
            # If something was contained, it will no longer be
            for frame_id in range(end_frame, self.total_frames + 1):
                self.contains[obj][frame_id] = None

    def get_dict(self):
        """
        Return the record in a human readable dictionary format
        """
        res = {}
        for ob, intervals in self.timeline.items():
            res[ob.name] = [
                self.human_readable_interval(interval) for interval
                in intervals]
        return res

    def human_readable_interval(self, interval):
        return (
            interval[0].__name__,
            interval[1].name if interval[1] else None,
            interval[2], interval[3])

    def was_contained(self, ob1, ob2, frame_id):
        """ Return true/false, based on whether ob2 was contained in ob1. """
        if ob1 is None:
            return False
        if ob1 == ob2:
            return True
        return self.was_contained(self.contains[ob1][frame_id], ob2, frame_id)


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')

