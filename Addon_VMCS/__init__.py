bl_info = {
    "name": "Tools",
    "author": "Hyuck Sang Lee, Multi-dimensional Insight Lab, Yonsei University",
    "version": (2023, 2, 27),
    "blender": (3, 3, 2),
    "location": "Viewport > Right panel",
    "description": "Virtual Multi Camera Studio MDI Tools",
    "category": "MDI"}
    
    
import bpy
import bmesh
from mathutils import Vector
from math import radians
import numpy as np
import os
import math

import time
import mathutils
import pickle as pkl
import scipy.io

import cv2
import random



from bpy.props import ( BoolProperty, EnumProperty, FloatProperty, PointerProperty ,IntProperty, StringProperty)
from bpy.types import ( PropertyGroup )



def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT, Rpytorch3d, Tpytorch3d = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT, Rpytorch3d, Tpytorch3d 


def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = mathutils.Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))
    R_cv2py = mathutils.Matrix(
        ((-1, 0,  0),
        (0, -1, 0),
        (0, 0, 1)))
        
#    R_bcam2cv = mathutils.Matrix(
#        ((1, 0,  0),
#        (0, 0, -1),
#        (0, 1, 0)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam



    # put into 3x4 matrix
    RTcv = mathutils.Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
        
        
    Rpytorch3d = R_bcam2cv@R_world2bcam@R_cv2py
    Tpytorch3d = T_world2cv
        
    return RTcv, Rpytorch3d, Tpytorch3d 


def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = mathutils.Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

def get_date():
    now = time.localtime()
    s = "%04d_%02d_%02d_%02d_%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    return s

  
def save_RT(result_path, real = False): # M R L
    
    k = 0
    Pair_list = []
    for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
        k += 1
        location, rotation = cam.matrix_world.decompose()[0:2]
        Pair_list.append(location)
        
    print(Pair_list)
    a = np.zeros(k)
    

    with open(result_path +'Calib/'+ "pair.txt", 'w') as f:
        f.write(str(k))
        for ii in range(k):
            f.write("\n"+str(ii) + "\n"+str(k-1)+" ")
            for j in range(k):
                a[j] = np.sum((np.array(Pair_list[j]) - np.array(Pair_list[ii]))**2)
                b = a.argsort()
                
            for h in range(k-1):
                f.write(str(b[h+1]) + " 100.0 ")
            print(b)
        f.close()    
            
    PP = np.zeros((k,3,4))
    KK = np.zeros((k,3,3))
    RRTT = np.zeros((k,3,4))

    for i, cam in enumerate([obj for obj in bpy.data.objects if obj.type == 'CAMERA']):
        bpy.context.scene.camera = cam
        P, K, RT, Rpytorch3d, Tpytorch3d  = get_3x4_P_matrix_from_blender(cam)      
            
        
        if real:
            directory = os.path.dirname(bpy.data.filepath)
            calib_file = bpy.context.window_manager.rmcs_tool.calib_file
            calibration_file = scipy.io.loadmat(directory +"/"+calib_file, struct_as_record=True)
            
            intrinsic_m = calibration_file['calibration']['CameraParameters'][0][0][0]

            K = intrinsic_m[i][0][0][0]
        PP[i] = P
        KK[i] = K
        RRTT[i] = RT
        
        with open(result_path +'Calib/'+ format(i,"08")+"_cam.txt", 'w') as f:
            f.write("extrinsic\n")
            f.write(str(round(RT[0][0], 6)) + ' ' + str(round(RT[0][1], 6)) + ' ' + str(round(RT[0][2], 6)) + ' ' + str(round(RT[0][3], 6)) + '\n')
            f.write(str(round(RT[1][0], 6)) + ' ' + str(round(RT[1][1], 6)) + ' ' + str(round(RT[1][2], 6)) + ' ' + str(round(RT[1][3], 6)) + '\n')
            f.write(str(round(RT[2][0], 6)) + ' ' + str(round(RT[2][1], 6)) + ' ' + str(round(RT[2][2], 6)) + ' ' + str(round(RT[2][3], 6)) + '\n')
            f.write("0.0 0.0 0.0 1.0\n\nintrinsic\n")
            f.write(str(K[0][0]) + ' ' + str(K[0][1]) + ' ' + str(K[0][2]) + '\n')
            f.write(str(K[1][0]) + ' ' + str(K[1][1]) + ' ' + str(K[1][2]) + '\n')
            f.write(str(K[2][0]) + ' ' + str(K[2][1]) + ' ' + str(K[2][2]) + '\n')
            
            f.write("\n900 1.5")
            f.close()    
        
        np.save(result_path +'Calib/pytorch3d_R'+format(i,'02')+'.npy',np.array(Rpytorch3d,dtype = np.float32))
        np.save(result_path +'Calib/pytorch3d_T'+format(i,'02')+'.npy',np.array(Tpytorch3d,dtype = np.float32))
        np.save(result_path +'Calib/pytorch3d_K'+format(i,'02')+'.npy',np.array(K,dtype = np.float32))
        
        
    np.save(result_path +'Calib/Camera_matrix_RT_3X4_'+format(k,'02')+'.npy',RRTT)
    np.save(result_path +'Calib/Camera_matrix_P_'+format(k,'02')+'.npy',PP)
    np.save(result_path +'Calib/Camera_matrix_K_'+format(k,'02')+'.npy',KK)
    dict = {"RT": RRTT, "K": KK,"P": PP}
    scipy.io.savemat(result_path + 'Camera_matrix.mat', dict)


def update_endframe(self,context):
    bpy.context.scene.frame_end = self.end_frame

def remove_all_cameras():
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()
    for cam in bpy.data.cameras:
        bpy.data.cameras.remove(cam)

class CameraAdd(bpy.types.Operator):
    bl_idname = "scene.camera_add"
    bl_label = "Add"
    bl_description = ("Add Camera of selected type to scene")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if "Camera Collection" not in bpy.data.collections:
    # If it doesn't, create the collection
            camera_collection = bpy.data.collections.new("Camera Collection")
            bpy.context.scene.collection.children.link(camera_collection)
            
        color = context.window_manager.vmcs_tool.color_type
        stereo = context.window_manager.vmcs_tool.stereo_type
        focal = context.window_manager.vmcs_tool.focal
        target = context.window_manager.vmcs_tool.target
        disparity = context.window_manager.vmcs_tool.disparity
            
        if stereo:
            module = bpy.data.objects.new("Stereo Module", None)
            module.location = bpy.context.scene.cursor.location
            module.constraints.new(type='TRACK_TO')
            module.constraints["Track To"].target = bpy.data.objects[target]
            
            
            cam1 = bpy.data.cameras.new("Camera"+ color)
            cam1.lens = focal
            cam2 = bpy.data.cameras.new("Camera"+ color)
            cam2.lens = focal
            
            cam1_obj = bpy.data.objects.new("Camera"+ color, cam1)
            cam2_obj = bpy.data.objects.new("Camera"+ color, cam2)
            cam1_obj.parent = bpy.data.objects[module.name]
            cam1_obj.location = (-disparity/2,0,0)
            
            cam2_obj.parent = bpy.data.objects[module.name]
            cam2_obj.location = (disparity/2,0,0)
            
            bpy.data.collections['Camera Collection'].objects.link(module)        
            bpy.data.collections['Camera Collection'].objects.link(cam1_obj)
            bpy.data.collections['Camera Collection'].objects.link(cam2_obj)

            
            
            
        else:
            cam = bpy.data.cameras.new("Camera"+ color)
            cam.lens = focal
            cam_obj = bpy.data.objects.new("Camera"+ color, cam)
            cam_obj.location = bpy.context.scene.cursor.location
            cam_obj.constraints.new(type='TRACK_TO')
            cam_obj.constraints["Track To"].target = bpy.data.objects[target]
            
            bpy.data.collections['Camera Collection'].objects.link(cam_obj)
            
        return {'FINISHED'}


class PG_VMCSProperties(PropertyGroup):

    color_type: EnumProperty(
        name = "Color",
        description = "Camera Color type",
        items = [ ("RGB", "RGB", ""), ("IR", "IR", "") ]
    )
    
    
    stereo_type: BoolProperty(
        name = "Stereo",
        description = "Stereo type",
        default = False
    )
    focal: FloatProperty(
        name = "Focal",
        description = "Camera Focal",
        default = 16.0
    )
    

    disparity: FloatProperty(
        name = "Disparity",
        description = "Stereo Camera Disparity",
        default = 0.1,
        min = 0.0,
        max = 1
    )
    
    
    target: EnumProperty(
        name = "Target",
        description = "Camera Target",
        items = [ ("Target.001", "Target.001", ""), ("Target.002", "Target.002", ""), ("Target.003", "Target.003", "") ]
        
        
    )
    
class ResetCamera(bpy.types.Operator):
    bl_idname = "scene.camera_reset"
    bl_label = "reset"
    
    bl_description = ("Delete All Cameras")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        remove_all_cameras()
    
        return {'FINISHED'}

class RTKProperties(PropertyGroup):

    type: EnumProperty(
        name = "Type",
        description = "Camera coordinate type",
        items = [ ("OpenCV", "OpenCV", ""), ("Pytorch 3D", "Pytorch 3D", "") ]
    )
    
    
    real: BoolProperty(
        name = "Use file data",
        description = "Use file intrinsic data",
        default = False
    )
    
     
class ExtractCameraMatrix(bpy.types.Operator):
    bl_idname = "scene.extract_camera_matrix"
    bl_label = "rt"
    bl_description = ("Extract Camera Matrix")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        
        type = context.window_manager.rt_tool.type
        real = context.window_manager.rt_tool.real

        directory = os.path.dirname(bpy.data.filepath)
        if not os.path.exists(directory+'/calib/'):
                os.makedirs(directory+'/calib/')
        if not os.path.exists(directory+'/calib/Calib/'):        
                os.makedirs(directory+'/calib/Calib/')
        save_RT(directory+'/calib/', real)
    
        return {'FINISHED'}    

class ShowCameraMatrix(bpy.types.Operator):
    bl_idname = "scene.show_camera_matrix"
    bl_label = "cam matrix"
    bl_description = ("Show Camera Matrix")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        
        calib_file = context.window_manager.rmcs_tool.calib_file

        calibration_file = scipy.io.loadmat(directory + '/'+calib_file, struct_as_record=True)

        intrinsic_m = calibration_file['calibration']['CameraParameters'][0][0][0]

        rotation_m = calibration_file['calibration']['ExtR'][0][0][0][:]

        trans_m = calibration_file['calibration']['ExtT'][0][0][0][:]
        
        name = bpy.context.selected_objects[0].name
        print(name + ' Intrinsic Information')
        print(intrinsic_m[int(name[11:13])][0][0][0])
    
        return {'FINISHED'}    
    
    
class RealCameraLoad(bpy.types.Operator):
    bl_idname = "scene.realcamera_add"
    bl_label = "Load"
    bl_description = ("Load Real Cameras from Calibration Matrix")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if "Camera Collection" not in bpy.data.collections:
    # If it doesn't, create the collection
            camera_collection = bpy.data.collections.new("Camera Collection")
            bpy.context.scene.collection.children.link(camera_collection)
        
        directory = os.path.dirname(bpy.data.filepath)
        calib_file = context.window_manager.rmcs_tool.calib_file
        sensor_size = context.window_manager.rmcs_tool.sensor_size
        
        remove_all_cameras()
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1024
        width = bpy.context.scene.render.resolution_x
        height = bpy.context.scene.render.resolution_y

        calibration_file = scipy.io.loadmat(directory + '/'+calib_file, struct_as_record=True)

        intrinsic_m = calibration_file['calibration']['CameraParameters'][0][0][0]

        rotation_m = calibration_file['calibration']['ExtR'][0][0][0][:]

        trans_m = calibration_file['calibration']['ExtT'][0][0][0][:]

        len_camera = len(trans_m)

        for i in range(len_camera):
            #img = bpy.data.images.load(directory+'/background/CAM'+format(i+1,'02')+'_00001'+'.png')
            img = bpy.data.images.load(directory+'/background/'+format(i,'08')+'.png')
            #bpy.ops.object.camera_add(enter_editmode=True, align='VIEW', location=trans_m[i]/1000, rotation= rotationMatrixToEulerAngles(np.transpose(rotation_m[i])), scale=(1, 1, 1))
            cam = bpy.data.cameras.new("Real Camera"+format(i,'02'))
            cam_obj = bpy.data.objects.new("Real Camera"+format(i,'02'), cam)
            
            
            blender_camera_rotation, blender_camera_location = extrinsic_to_blender(rotation_m[i].astype(np.float), trans_m[i])
            
            
            
            
            cam_obj.location = blender_camera_location/1000
            cam_obj.rotation_euler = blender_camera_rotation
            #print(intrinsic_m[i])
            #cam.lens = (intrinsic_m[i][0][0][0][0,0] + intrinsic_m[i][0][0][0][1,1]) / 2 * cam.sensor_width/ width
            
            cam.lens = intrinsic_m[i][0][0][0][0,0] * cam.sensor_width/ width
            
            
            #print(intrinsic_m[i][0][0][0][0,0] * cam.sensor_width/ width)
            #print(intrinsic_m[i][0][0][0][1,1] * cam.sensor_height/ height)
            cam.sensor_fit = 'HORIZONTAL' 
            cam.sensor_width = 36
            #cam.sensor_height = 14.9
            
            print(intrinsic_m[i][0][0][0])
            cam.shift_x = -(intrinsic_m[i][0][0][0][0,2] - width/2) / (width)
            cam.shift_y = (intrinsic_m[i][0][0][0][1,2] - height/2) / (width)
            #this is right do not use height
            
            
            cam.show_background_images = True
            bg = cam.background_images.new()
            bg.image = img
            #bg.rotation = 1.5708
            bg.frame_method = 'FIT'
            #bg.scale = 1.33000
            bg.alpha = 0.8
            bpy.data.collections['Camera Collection'].objects.link(cam_obj)

        bpy.context.scene.use_nodes = True
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.use_nodes = True
    
        return {'FINISHED'}


class AlignCamera(bpy.types.Operator):
    bl_idname = "scene.aligncamera"
    bl_label = "Align"
    bl_description = ("Align Real Cameras World coordinate")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        mid_point = [0,0,0]
        k = 0
        for i, cam in enumerate([obj for obj in bpy.data.objects if obj.type == 'CAMERA']):
            
            mid_point += np.array(cam.location)
            k += 1
        mid_point = mid_point/k 
        print(mid_point)
        
        for i, cam in enumerate([obj for obj in bpy.data.objects if obj.type == 'CAMERA']):
            cam.location = np.array(cam.location) - mid_point
        
        
        bpy.ops.object.select_all(action='DESELECT')
        for i, cam in enumerate([obj for obj in bpy.data.objects if obj.type == 'CAMERA']):
            cam.select_set(True)
            if i == 0:
                ref = np.array(cam.rotation_euler)
                rot_x = ref[0] -  1.570796327
        bpy.ops.transform.rotate(value=rot_x, orient_axis='X', orient_type='GLOBAL')

        
        
    
        return {'FINISHED'}




class PG_RMCSProperties(PropertyGroup):

    calib_file: bpy.props.StringProperty(
        name = "Calib_file",
        description = "Path",
        default = "calibration.mat"
    )
    
    sensor_size: FloatProperty(
        name = "Sensor Size",
        description = "Sensor Size",
        default = 36.0
    )
    


class VMCS_PT_Tools(bpy.types.Panel):
    bl_label = "Camera Setting"
    bl_category = "VMCS"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"


    def draw(self, context):

        try:
            ob = context.object
            mode = ob.mode
            name = ob.name
        except:
            ob = None
            mode = None
            name = ''

        layout = self.layout
        col = layout.column(align=True)
#        row = col.row(align=True)
#        row.operator("ed.undo", icon='LOOP_BACK')
#        row.operator("ed.redo", icon='LOOP_FORWARDS')
#        col.separator()

        row = col.row(align=True)
        col.label(text="Add Virtual Camera")
        col.prop(context.window_manager.vmcs_tool, "color_type")
        col.separator()
        col.prop(context.window_manager.vmcs_tool, "focal")
        col.separator()
        col.prop(context.window_manager.vmcs_tool, "target")
        
        col.separator()
        col.separator()
        split = col.split(factor=0.3, align=True)
        split.prop(context.window_manager.vmcs_tool, "stereo_type")
        split.prop(context.window_manager.vmcs_tool, "disparity")
        
        col.operator("scene.camera_add", text="Add Camera")
        col.separator()
        col.separator()
        col.label(text="Sink Real data")
        col.prop(context.window_manager.rmcs_tool, "calib_file")
        col.separator()
        col.prop(context.window_manager.rmcs_tool, "sensor_size")
        col.separator()
        col.operator("scene.realcamera_add", text="Load Real Camera")
        col.separator()
        col.operator("scene.show_camera_matrix", text="Show Intrinsic Active Camera")
        col.separator()
        col.operator("scene.aligncamera", text="Align Camera")
        col.separator()
        col.separator()
        col.label(text="Extract Camera Matrixs")
        col.prop(context.window_manager.rt_tool, "type")
        col.separator()
        col.prop(context.window_manager.rt_tool, "real")
        col.separator()
        col.operator("scene.extract_camera_matrix", text="Extract Camera Matrix")
        col.separator()
        col.separator()
        col.label(text="Reset All Cameras")
        col.operator("scene.camera_reset", text="Reset All Camera")






    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)




        
        
#        col.label(text="Blend Shapes:")
#        row = col.row(align=True)
#        split = row.split(factor=0.75, align=True)
#        split.operator("object.flame_random_shapes", text="Random head shape")
#        split.operator("object.flame_reset_shapes", text="Reset")
#        col.separator()
#        row = col.row(align=True)
#        split = row.split(factor=0.75, align=True)
#        split.operator("object.flame_random_expressions", text="Random facial expression")
#        split.operator("object.flame_reset_expressions", text="Reset")
#        col.separator()

#        col.label(text="Joints:")
#        col.operator("object.flame_update_joint_locations", text="Update joint locations")
#        col.separator()

#        col.label(text="Pose:")
#        col.prop(context.window_manager.flame_tool, "flame_corrective_poseshapes")
#        col.separator()
#        col.operator("object.flame_set_poseshapes", text="Set poseshapes for current pose")
#        col.separator()
#        col.separator()
#        col.prop(context.window_manager.flame_tool, "flame_neck_yaw", slider=True)
#        col.prop(context.window_manager.flame_tool, "flame_neck_pitch", slider=True)
#        col.prop(context.window_manager.flame_tool, "flame_jaw", slider=True)
#        row = col.row(align=True)
#        split = row.split(factor=0.75, align=True)
#        split.operator("object.flame_write_pose", text="Write pose to console")
#        split.operator("object.flame_reset_pose", text="Reset")
#        col.separator()

#        col.label(text="Misc Tools:")
#        row = col.row(align=True)
#        row.operator("object.flame_close_mesh", text="Close mesh")
#        row.operator("object.flame_restore_mesh", text="Open mesh")
#        col.separator()
#        export_button = col.operator("export_scene.obj", text="Export OBJ [mm]", icon='EXPORT')
#        export_button.global_scale = 1000.0
#        export_button.use_selection = True
#        col.separator()
#        (year, month, day) = bl_info["version"]
#        col.label(text="Version: %s-%s-%s" % (year, month, day))

classes = [
    PG_VMCSProperties,
    PG_RMCSProperties,
    RTKProperties,
    CameraAdd,
    RealCameraLoad,
    AlignCamera,
    ResetCamera,
    ExtractCameraMatrix,
    ShowCameraMatrix,
#    FlameRandomShapes,
#    FlameResetShapes,
#    FlameRandomExpressions,
#    FlameResetExpressions,
#    FlameUpdateJointLocations,
#    FlameSetPoseshapes,
#    FlameResetPoseshapes,
#    FlameWritePose,
#    FlameResetPose,
#    FlameCloseMesh,
#    FlameRestoreMesh,
    VMCS_PT_Tools
]

def register():
    from bpy.utils import register_class
    #print(bpy.utils.resource_path('USER'))
    
    for cls in classes:
        bpy.utils.register_class(cls)

    # Store properties under WindowManager (not Scene) so that they are not saved in .blend files and always show default values after loading
    bpy.types.WindowManager.vmcs_tool = PointerProperty(type=PG_VMCSProperties)
    bpy.types.WindowManager.rmcs_tool = PointerProperty(type=PG_RMCSProperties)
    bpy.types.WindowManager.rt_tool = PointerProperty(type=RTKProperties)
    
    
def unregister():
    from bpy.utils import unregister_class
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.WindowManager.flame_tool

if __name__ == "__main__":
    
    register()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    