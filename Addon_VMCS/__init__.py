bl_info = {
    "name": "Tools",
    "author": "Hyuck Sang Lee, Multi-dimensional Insight Lab, Yonsei University",
    "version": (2023, 1, 30),
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
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT


def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = mathutils.Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

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
    RT = mathutils.Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT


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

def projection_mat_tmp(camera, render):
        # Get the two components to calculate M
    modelview_matrix = camera.matrix_world.inverted()
    projection_matrix = camera.calc_matrix_camera(
        bpy.data.scenes["Scene"].view_layers["View Layer"].depsgraph,
        x = render.resolution_x,
        y = render.resolution_y,
        scale_x = render.pixel_aspect_x,
        scale_y = render.pixel_aspect_y,
    )
    P = projection_matrix @ modelview_matrix
    
    return P



def get_date():
    now = time.localtime()
    s = "%04d_%02d_%02d_%02d_%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    return s

def euler_to_rotVec(yaw, pitch, roll):
    # compute the rotation matrix
    Rmat = euler_to_rotMat(yaw, pitch, roll)
    
    theta = math.acos(((Rmat[0, 0] + Rmat[1, 1] + Rmat[2, 2]) - 1) / 2)
    sin_theta = math.sin(theta)
    if sin_theta < 1e-6:
        rx, ry, rz = 0.0, 0.0, 0.0
    else:
        multi = 1 / (2 * math.sin(theta))
        rx = multi * (Rmat[2, 1] - Rmat[1, 2]) * theta
        ry = multi * (Rmat[0, 2] - Rmat[2, 0]) * theta
        rz = multi * (Rmat[1, 0] - Rmat[0, 1]) * theta
    return rx, ry, rz

def euler_to_rotMat(yaw, pitch, roll):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]])
    # R = RzRyRx
    rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    return rotMat

  
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
        P, K, RT = get_3x4_P_matrix_from_blender(cam)      
            
        
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
            f.write(str(round(RT[0][0], 6)) + ' ' + str(round(RT[0][1], 6)) + ' ' + str(round(RT[0][2], 6)) + ' ' + str(round(1000*RT[0][3], 6)) + '\n')
            f.write(str(round(RT[1][0], 6)) + ' ' + str(round(RT[1][1], 6)) + ' ' + str(round(RT[1][2], 6)) + ' ' + str(round(1000*RT[1][3], 6)) + '\n')
            f.write(str(round(RT[2][0], 6)) + ' ' + str(round(RT[2][1], 6)) + ' ' + str(round(RT[2][2], 6)) + ' ' + str(round(1000*RT[2][3], 6)) + '\n')
            f.write("0.0 0.0 0.0 1.0\n\nintrinsic\n")
            f.write(str(K[0][0]) + ' ' + str(K[0][1]) + ' ' + str(K[0][2]) + '\n')
            f.write(str(K[1][0]) + ' ' + str(K[1][1]) + ' ' + str(K[1][2]) + '\n')
            f.write(str(K[2][0]) + ' ' + str(K[2][1]) + ' ' + str(K[2][2]) + '\n')
            
            f.write("\n900 1.5")
            f.close()    
        
    np.save(result_path +'Calib/Camera_matrix_RT_3X4_'+format(k,'02')+'.npy',RRTT)
    np.save(result_path +'Calib/Camera_matrix_P_'+format(k,'02')+'.npy',PP)
    np.save(result_path +'Calib/Camera_matrix_K_'+format(k,'02')+'.npy',KK)
    dict = {"RT": RRTT, "K": KK,"P": PP}
    scipy.io.savemat(result_path + 'Camera_matrix.mat', dict)




def extrinsic_to_blender(rotation, translation):

    print(rotation.dtype)
    adjustment_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rotataion_matrix, _ = cv2.Rodrigues(rotation)
    blender_camera_location = -rotataion_matrix.T @ translation
    blender_camera_rotation = rotationMatrixToEulerAngles(
        rotataion_matrix.T @ adjustment_mat
    )
    return blender_camera_rotation, blender_camera_location    
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-4

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    #print(isRotationMatrix(R))
    
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])
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




def change_name(result_path, file_name, n, n4, n9, n3, n12):
    if n4:    
        file_oldname = os.path.join(result_path + "/Depth", "Image"+ format(n,"04")+".png")
        file_newname_newfile = os.path.join(result_path + "/Depth", file_name + "_depth.png")
        os.rename(file_oldname, file_newname_newfile)
    if n9:
        file_oldname = os.path.join(result_path + "/Noise", "Image"+ format(n,"04")+".png")
        file_newname_newfile = os.path.join(result_path + "/Noise", file_name + ".png")
        os.rename(file_oldname, file_newname_newfile)
    if n3:
        img = cv2.imread(result_path+ "/Mask/" + "Image"+ format(n,"04") +'.png',cv2.IMREAD_UNCHANGED)
        ret, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        cv2.imwrite(result_path+'Mask/'+file_name +'_mask.png',mask)
    if n12:
        file_oldname = os.path.join(result_path + "/Diffuse", "Image"+ format(n,"04")+".png")
        file_newname_newfile = os.path.join(result_path + "/Diffuse", file_name + ".png")
        os.rename(file_oldname, file_newname_newfile)
        

class DataSaverMainPanel(bpy.types.Panel):
    bl_label = "Data Saver"
    bl_idname = "Data_Saver_MAINPANEL"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Data Saver"

    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        row.label(text= "Data Save Setting")
        
        row = layout.row()
        row.operator('data_saver.set_operator')
        row = layout.row()
        row.operator('composite.set_operator')

class CompositeSetting_PT_CONTROL(bpy.types.Operator):
    bl_label = "Composite Setting Load"
    bl_idname = 'composite.set_operator'

    def execute(self, context):
        bpy.context.scene.use_nodes = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_diffuse_color = True

        # Get the compositing node tree in the target file
        compositing_tree = bpy.context.scene.node_tree


        file_output_node = compositing_tree.nodes.new("CompositorNodeOutputFile")
        file_output_node.name = "File Output"
        file_output1_node = compositing_tree.nodes.new("CompositorNodeOutputFile")
        file_output1_node.name = "File Output.001"
        file_output2_node = compositing_tree.nodes.new("CompositorNodeOutputFile")
        file_output2_node.name = "File Output.002"
        file_output3_node = compositing_tree.nodes.new("CompositorNodeOutputFile")
        file_output3_node.name = "File Output.003"

        blur_node = compositing_tree.nodes.new("CompositorNodeBlur")
        
        defocus_node = compositing_tree.nodes.new("CompositorNodeDefocus")
        
        map_node = compositing_tree.nodes.new("CompositorNodeMapRange")
        normalize_node = compositing_tree.nodes.new("CompositorNodeNormalize")
        map_node = compositing_tree.nodes.new("CompositorNodeGamma")
        color_ramp_node = compositing_tree.nodes.new("CompositorNodeValToRGB")
        setalpha_node = compositing_tree.nodes.new("CompositorNodeSetAlpha")

        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["Gamma"].inputs[0])
        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[1],bpy.data.scenes['Scene'].node_tree.nodes["File Output.002"].inputs[0])
        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[2],bpy.data.scenes['Scene'].node_tree.nodes["Map Range"].inputs[0])
        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[2],bpy.data.scenes['Scene'].node_tree.nodes["Normalize"].inputs[0])
        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[3],bpy.data.scenes['Scene'].node_tree.nodes["Set Alpha"].inputs[0])
        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Set Alpha"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["File Output.003"].inputs[0])
        
        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Normalize"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["ColorRamp"].inputs[0])
        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["ColorRamp"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["Defocus"].inputs[1])
        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Gamma"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["Defocus"].inputs[0])
        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Gamma"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["Blur"].inputs[0])
        
        bpy.data.scenes["Scene"].node_tree.nodes["Map Range"].inputs[2].default_value = 3
        bpy.data.scenes["Scene"].node_tree.nodes['File Output'].format.compression = 50
        bpy.data.scenes["Scene"].node_tree.nodes['File Output.001'].format.compression = 50
        bpy.data.scenes["Scene"].node_tree.nodes['File Output.003'].format.compression = 50
        bpy.data.scenes["Scene"].node_tree.nodes['File Output.002'].format.compression = 100
        
        
        # Save the target blend file
        bpy.ops.wm.save_mainfile()
        
        return{'FINISHED'}   

class DataSaveSetting_PT_CONTROL(bpy.types.Operator):
    bl_label = "Data Save"
    bl_idname = 'data_saver.set_operator'



    number1: bpy.props.IntProperty(name= "View point (0 is Multi)", default = 0)
    number11: bpy.props.BoolProperty(name= "Image", default = 1)
    
    number2: bpy.props.BoolProperty(name= "Camera matrix", default = 1)
    
    number3: bpy.props.BoolProperty(name= "Mask", default = 1)
    number4: bpy.props.BoolProperty(name= "Depth", default = 1)
    
    number8: bpy.props.BoolProperty(name= "Obj", default = 1)
    
    number12:bpy.props.BoolProperty(name= "Diffuse", default = 0) 
    number10: bpy.props.BoolProperty(name= "With HDR background", default = 0)
    
    number9: bpy.props.BoolProperty(name= "Noise image", default = 0)
    number5: bpy.props.IntProperty(name= "Start Frame", default = 0)
    
    
    number6: bpy.props.IntProperty(name= "Number of Frame", default = 0)

    number7: bpy.props.StringProperty(name= "Folder Name", default = "Virtual_result")


    
    
    def execute(self, context):
        path = os.path.dirname(bpy.data.filepath) +'/'
        n1 = self.number1
        n2 = self.number2
        n3 = self.number3
        n4 = self.number4
        n5 = self.number5
        n6 = self.number6
        n7 = self.number7
        n8 = self.number8
        n9 = self.number9
        n10 = self.number10
        n11 = self.number11
        n12 = self.number12
        
        
        if n1 == 0: #multi view
            date = get_date()
            result_path = path + "Rendered_Results/" + n7 + "_multi_view_" + date + "/"
            if not os.path.exists(result_path):
                os.makedirs(result_path)
                if n8:
                    os.makedirs(result_path +'obj/') 
                if n2:
                    os.makedirs(result_path +'Calib/')         
                if n4:
                    os.makedirs(result_path +'Depth/')
                    bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Map Range"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["File Output"].inputs[0])
                    bpy.data.scenes['Scene'].node_tree.nodes["File Output"].base_path = result_path + "/Depth" 
                    bpy.data.scenes['Scene'].node_tree.nodes["File Output"].format.color_mode = 'BW'
                    bpy.data.scenes['Scene'].node_tree.nodes["File Output"].format.color_depth = '16'
                else:
                    bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Map Range"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["File Output"].inputs[0]) 
                    l = bpy.data.scenes['Scene'].node_tree.nodes["Map Range"].outputs[0].links[0]
                    bpy.data.scenes["Scene"].node_tree.links.remove(l)
                    
                if n9:
                    os.makedirs(result_path +'Noise/')     
                    bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Defocus"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["File Output.001"].inputs[0])
                    bpy.data.scenes['Scene'].node_tree.nodes["File Output.001"].base_path = result_path + "/Noise"
                    bpy.data.scenes['Scene'].node_tree.nodes["File Output.001"].format.color_mode = 'RGBA'
                    bpy.data.scenes['Scene'].node_tree.nodes["File Output.001"].format.color_depth = '8'  
            
                else:
                    bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Defocus"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["File Output.001"].inputs[0]) 
                    l = bpy.data.scenes['Scene'].node_tree.nodes["Defocus"].outputs[0].links[0]
                    bpy.data.scenes["Scene"].node_tree.links.remove(l)
                    
                if n10:
                    os.makedirs(result_path +'With_background/')     
                
                
                if n3:
                    os.makedirs(result_path +'Mask/')
                    bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[1],bpy.data.scenes['Scene'].node_tree.nodes["File Output.002"].inputs[0])
                    bpy.data.scenes['Scene'].node_tree.nodes["File Output.002"].base_path = result_path + "/Mask" 
                    bpy.data.scenes['Scene'].node_tree.nodes["File Output.002"].format.color_mode = 'BW'
                    bpy.data.scenes['Scene'].node_tree.nodes["File Output.002"].format.color_depth = '8'
                else:
                    bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[1],bpy.data.scenes['Scene'].node_tree.nodes["File Output.002"].inputs[0]) 
                    l = bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[1].links[0]
                    bpy.data.scenes["Scene"].node_tree.links.remove(l)
                    
                if n12:
                    os.makedirs(result_path +'Diffuse/')
                    bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[3],bpy.data.scenes['Scene'].node_tree.nodes["Set Alpha"].inputs[0])
                    bpy.data.scenes['Scene'].node_tree.nodes["File Output.003"].base_path = result_path + "/Diffuse" 
                    bpy.data.scenes['Scene'].node_tree.nodes["File Output.003"].format.color_mode = 'RGBA'
                    bpy.data.scenes['Scene'].node_tree.nodes["File Output.003"].format.color_depth = '8'  

                else:
                    bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[3],bpy.data.scenes['Scene'].node_tree.nodes["Set Alpha"].inputs[0]) 
                    l = bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[3].links[0]
                    bpy.data.scenes["Scene"].node_tree.links.remove(l)
            
                    
            if n2:       
                save_RT(result_path) 

            bpy.data.scenes['Scene'].frame_current = n5   
                                   
            
            for i in range(n6 + 1):
                bpy.data.scenes['Scene'].frame_current= n5 + i
                
             
                if n8:
                    target_file = os.path.join(result_path+ 'obj/', 'Position'+format(i,'06')+'.obj')
                    bpy.ops.export_scene.obj(filepath=target_file, use_selection=True, use_materials=False, axis_forward = 'Y' , axis_up = 'Z')
                
                for k, cam in enumerate([obj for obj in bpy.data.objects if obj.type == 'CAMERA']):
                     
                    bpy.context.scene.camera = cam
                    bpy.context.scene.render.film_transparent = True #no background (hdr)
                    
                    
                    if n4:
                        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Map Range"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["File Output"].inputs[0])
                    else:
                        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Map Range"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["File Output"].inputs[0]) 
                        l = bpy.data.scenes['Scene'].node_tree.nodes["Map Range"].outputs[0].links[0]
                        bpy.data.scenes["Scene"].node_tree.links.remove(l)
                    
                    if n9:   
                        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Defocus"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["File Output.001"].inputs[0]) 
                        gamma_val = random.gauss(1.0, 0.5)
                        if gamma_val <0.7:
                            gamma_val = -gamma_val + 2
                        
                        
                        defocus1 = 1-abs(random.gauss(0.0, 0.3))
                        defocus2 = abs(random.gauss(0.0, 5))
                        bpy.data.scenes["Scene"].node_tree.nodes["Gamma"].inputs[1].default_value = gamma_val
                        bpy.data.scenes["Scene"].node_tree.nodes["ColorRamp"].color_ramp.elements[1].position = defocus1
                        bpy.data.scenes["Scene"].node_tree.nodes["Defocus"].z_scale = defocus2
                    else:
                        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Defocus"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["File Output.001"].inputs[0]) 
                        l = bpy.data.scenes['Scene'].node_tree.nodes["Defocus"].outputs[0].links[0]
                        bpy.data.scenes["Scene"].node_tree.links.remove(l)
                    
                    if n3:
                        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[1],bpy.data.scenes['Scene'].node_tree.nodes["File Output.002"].inputs[0])
                    else:
                        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[1],bpy.data.scenes['Scene'].node_tree.nodes["File Output.002"].inputs[0]) 
                        l = bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"].outputs[1].links[0]
                        bpy.data.scenes["Scene"].node_tree.links.remove(l)
                    
                        

                    
                    file_name = "Position"+ format(i,'06')+"_CAM" +format(k,'02')
                    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
                    bpy.context.scene.render.image_settings.color_depth = '8'
                    bpy.context.scene.render.filepath = os.path.join(result_path, file_name)
                    bpy.ops.render.render(write_still=n11)
                    change_name(result_path, file_name, n5 + i, n4, n9, n3, n12)
                    if n10:
                        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Map Range"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["File Output"].inputs[0]) 
                        l = bpy.data.scenes['Scene'].node_tree.nodes["Map Range"].outputs[0].links[0]
                        bpy.data.scenes["Scene"].node_tree.links.remove(l)
                        
                        bpy.data.scenes["Scene"].node_tree.links.new(bpy.data.scenes['Scene'].node_tree.nodes["Defocus"].outputs[0],bpy.data.scenes['Scene'].node_tree.nodes["File Output.001"].inputs[0]) 
                        l = bpy.data.scenes['Scene'].node_tree.nodes["Defocus"].outputs[0].links[0]
                        bpy.data.scenes["Scene"].node_tree.links.remove(l)
                        
                        
                        
                        bpy.context.scene.render.film_transparent = False #With background 
                        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
                        bpy.context.scene.render.filepath = os.path.join(result_path+ 'With_background/', file_name)
                        bpy.ops.render.render(write_still=True)
            
                if os.path.exists(result_path + "/Mask/Image" + format(n5 + i,"04")+".png"):         
                    os.remove(result_path + "/Mask/Image" + format(n5 + i,"04")+".png")                    
                if os.path.exists(result_path + "/Diffuse/Image" + format(n5 + i,"04")+".png"):         
                    os.remove(result_path + "/Diffuse/Image" + format(n5 + i,"04")+".png")     
                    
                    
                        
                    
    

        
        return{'FINISHED'}

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
    VMCS_PT_Tools,
    DataSaverMainPanel,
    DataSaveSetting_PT_CONTROL,
    CompositeSetting_PT_CONTROL
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
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    