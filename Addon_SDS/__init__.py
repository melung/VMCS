bl_info = {
    "name": "Sythetic Data Saver",
    "author": "Hyuck Sang Lee, Multi-dimensional Insight Lab, Yonsei University",
    "version": (2023, 2, 27),
    "blender": (3, 3, 2),
    "location": "Viewport > Right panel",
    "description": "Virtual Multi Camera Studio MDI Tools",
    "category": "MDI"}

import bpy
import math
import time
import numpy as np
import os
import mathutils
import pickle as pkl
import scipy.io
import cv2
import random



def make_new_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
        return directory_path
    else:
        i = 1
        while True:
            numbered_directory_path = f"{directory_path}_{i}"
            if not os.path.exists(numbered_directory_path):
                os.makedirs(numbered_directory_path)
                print(f"Created directory: {numbered_directory_path}")
                break
            i += 1
        return numbered_directory_path

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
    


def change_name(result_path, file_name, n, depth_img, noise_img, mask_img, diffuse, mask_num):
    if depth_img:    
        file_oldname = os.path.join(result_path + "/Depth", "Image"+ format(n,"04")+".png")
        file_newname_newfile = os.path.join(result_path + "/Depth", file_name + "_depth.png")
        os.rename(file_oldname, file_newname_newfile)
    if noise_img:
        file_oldname = os.path.join(result_path + "/Noise", "Image"+ format(n,"04")+".png")
        file_newname_newfile = os.path.join(result_path + "/Noise", file_name + ".png")
        os.rename(file_oldname, file_newname_newfile)
    if mask_img:
        for i in range(mask_num):
            img = cv2.imread(result_path+ "/Mask"+format(i,"02")+"/" + "Image"+ format(n,"04") +'.png',cv2.IMREAD_UNCHANGED)
            ret, mask = cv2.threshold(img,250, 255, cv2.THRESH_BINARY)
            cv2.imwrite(result_path+'/Mask'+format(i,"02")+"/"+file_name +'_mask.png',mask)     
    if diffuse:
        file_oldname = os.path.join(result_path + "/Diffuse", "Image"+ format(n,"04")+".png")
        file_newname_newfile = os.path.join(result_path + "/Diffuse", file_name + ".png")
        os.rename(file_oldname, file_newname_newfile)
        

def InitCompositing(result_path, img, cam_mat, mask_img, mask_num, depth_img, obj, diffuse, hdr_img, noise_img, frames):
    scene = bpy.context.scene
    
    for node in scene.node_tree.nodes:
        if node.type != 'COMPOSITE':
            scene.node_tree.nodes.remove(node)
    
    # Create a new render layer
    input_node = scene.node_tree.nodes.new(type = 'CompositorNodeRLayers')
    
    # Set the render layer's settings
    bpy.context.scene.view_layers["ViewLayer"].use_pass_combined = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
       
    output_node = scene.node_tree.nodes['Composite']
    scene.node_tree.links.new(input_node.outputs[0], output_node.inputs[0])
          
        
    if obj:
        if not os.path.exists(result_path +'obj/'):
            os.makedirs(result_path +'obj/') 
            
    if cam_mat:
        if not os.path.exists(result_path +'Calib/'):
            os.makedirs(result_path +'Calib/')
               
    if depth_img:
        if not os.path.exists(result_path +'Depth/'):
            os.makedirs(result_path +'Depth/')
        
        map_range = scene.node_tree.nodes.new('CompositorNodeMapRange')
        # Set the node  depth properties
        map_range.inputs[1].default_value = 0.0 # input minimum
        map_range.inputs[2].default_value = 3.0 # input maximum
        map_range.inputs[3].default_value = 0.0 # output minimum
        map_range.inputs[4].default_value = 1.0 # output maximum
        
        depth_output = scene.node_tree.nodes.new('CompositorNodeOutputFile')
        depth_output.name = 'Depth Output'
        
        scene.node_tree.links.new(input_node.outputs[2],map_range.inputs[0])
        
        scene.node_tree.links.new(map_range.outputs[0],depth_output.inputs[0])
        
        depth_output.base_path = result_path + "/Depth" 
        depth_output.format.color_mode = 'BW'
        depth_output.format.color_depth = '16'

    if noise_img:
        if not os.path.exists(result_path +'Noise/'):
            os.makedirs(result_path +'Noise/')
        
        gamma_node = scene.node_tree.nodes.new('CompositorNodeGamma')
        
        noise_node = scene.node_tree.nodes.new('CompositorNodeDefocus')
        
        noise_output = scene.node_tree.nodes.new('CompositorNodeOutputFile')
        noise_output.name = 'Noise Output'
        
        scene.node_tree.links.new(input_node.outputs[0], gamma_node.inputs[0])
        
        scene.node_tree.links.new(gamma_node.outputs[0], noise_node.inputs[0])
        
        scene.node_tree.links.new(noise_node.outputs[0], noise_output.inputs[0])
        
        noise_output.base_path = result_path + "/Noise"
        noise_output.format.color_mode = 'RGBA'
        noise_output.format.color_depth = '8'
        
    if hdr_img:
        if not os.path.exists(result_path +'With_background/'):
            os.makedirs(result_path +'With_background/') 
            
    
    if mask_img:
        for i in range(mask_num):
            if not os.path.exists(result_path +'Mask'+format(i,"02")+'/'):
                os.makedirs(result_path +'Mask'+format(i,"02")+'/')
            
            
            mask_output = scene.node_tree.nodes.new('CompositorNodeOutputFile')
            mask_output.name = 'Mask Output.' +format(i,"02")
            
            mask_index = scene.node_tree.nodes.new('CompositorNodeIDMask')
            mask_index.index = i
            
            set_mask = scene.node_tree.nodes.new('CompositorNodeMath')
            set_mask.operation = 'MULTIPLY'
                 
            scene.node_tree.links.new(input_node.outputs[3],mask_index.inputs[0])
            
            scene.node_tree.links.new(mask_index.outputs[0],set_mask.inputs[1])
            scene.node_tree.links.new(input_node.outputs[1],set_mask.inputs[0])
            
            scene.node_tree.links.new(set_mask.outputs[0],mask_output.inputs[0])
            
            mask_output.base_path = result_path + "/Mask"+format(i,"02")
            mask_output.format.color_mode = 'BW'
            mask_output.format.color_depth = '8'
        
    if diffuse:
        if not os.path.exists(result_path +'Diffuse/'):
                os.makedirs(result_path +'Diffuse/')
        
        
        diffuse_output = scene.node_tree.nodes.new('CompositorNodeOutputFile')
        diffuse_output.name = 'Diffuse Output'
        
        scene.node_tree.links.new(input_node.outputs[4], diffuse_output.inputs[0])
        diffuse_output.base_path = result_path + "/Diffuse" 
        diffuse_output.format.color_mode = 'RGBA'
        diffuse_output.format.color_depth = '8'  
        

class DataSaverMainPanel(bpy.types.Panel):
    bl_label = "Synthetic Data Saver"
    bl_idname = "Data_Saver_MAINPANEL"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Synthetic Data Saver"

    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        row.label(text= "Data Save Setting")
        
        row = layout.row()
        row.operator('data_saver.set_operator')


class DataSaveSetting_PT_CONTROL(bpy.types.Operator):
    bl_label = "Data Save"
    bl_idname = 'data_saver.set_operator'
    
    root_path: bpy.props.StringProperty(name= "Path", default = "E:/")
    
    image: bpy.props.BoolProperty(name= "Image", default = True)
    
    camera_matrix: bpy.props.BoolProperty(name= "Camera matrix", default = 1)
    
    mask_image: bpy.props.BoolProperty(name= "Mask", default = 1)
    
    mask_number: bpy.props.IntProperty(name= "Mask Number", default = 1)
    
    depth_image: bpy.props.BoolProperty(name= "Depth", default = 1)
    
    object: bpy.props.BoolProperty(name= "Obj", default = 1)
    
    diffuse_image:bpy.props.BoolProperty(name= "Diffuse", default = 0) 
    hdr_background: bpy.props.BoolProperty(name= "With HDR background", default = 0)
    
    noise_image: bpy.props.BoolProperty(name= "Noise image", default = 0)
    frame_sel: bpy.props.BoolProperty(name= "Multi Frames", default = False)
    
    folder: bpy.props.StringProperty(name= "Folder Name", default = "Virtual_result")
    

    
    def execute(self, context):
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.use_nodes = True

        
        
        img = self.image
        
        cam_mat = self.camera_matrix
        
        mask_img = self.mask_image
        mask_num = self.mask_number
        
        depth_img = self.depth_image
        
        obj = self.object
        
        diffuse = self.diffuse_image
        hdr_img = self.hdr_background
        noise_img = self.noise_image
        
        frames = self.frame_sel
        
        result_root_path = self.root_path
        date = get_date()
        result_path = os.path.join(result_root_path,  self.folder)
        compsite_node_tree = bpy.context.scene.node_tree
        result_path = make_new_directory(result_path) +"/"

        InitCompositing(result_path, img, cam_mat, mask_img, mask_num, depth_img, obj, diffuse, hdr_img, noise_img, frames)
        
     
        if cam_mat:       
            save_RT(result_path) 

        total_frames = 0
        if frames:
            total_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start                        
        
        for i in range(total_frames + 1):
            if frames:
                bpy.context.scene.frame_current= bpy.context.scene.frame_start  + i
            
            if obj:
                target_file = os.path.join(result_path+ 'obj/', 'Position'+format(i,'06')+'.obj')
                bpy.ops.export_scene.obj(filepath=target_file, use_selection=True, use_materials=False, axis_forward = '-Y' , axis_up = 'Z')
            
            for k, cam in enumerate([obj for obj in bpy.data.objects if obj.type == 'CAMERA']):
                 
                bpy.context.scene.camera = cam
                bpy.context.scene.render.film_transparent = True #no background (hdr)
                
                if noise_img:   
                    gamma_val = random.gauss(1.0, 0.5)
                    if gamma_val <0.7:
                        gamma_val = -gamma_val + 2
                    defocus1 = 1-abs(random.gauss(0.0, 0.3))
                    defocus2 = abs(random.gauss(0.0, 5))
                    
                    compsite_node_tree.nodes["Gamma"].inputs[1].default_value = gamma_val
                    compsite_node_tree.nodes["Defocus"].z_scale = defocus2
        
        
                file_name = "Position"+ format(i,'06')+"_CAM" +format(k,'02')
                bpy.context.scene.render.image_settings.color_mode = 'RGBA'
                bpy.context.scene.render.image_settings.color_depth = '8'
                bpy.context.scene.render.filepath = os.path.join(result_path, file_name)
                
                bpy.ops.render.render(write_still=img)
                
                ##
                change_name(result_path, file_name, bpy.context.scene.frame_current, depth_img, noise_img, mask_img, diffuse, mask_num)
                if hdr_img:
                    #later 
                    
                    
                    
                    bpy.context.scene.render.film_transparent = False #With background 
                    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
                    bpy.context.scene.render.filepath = os.path.join(result_path+ 'With_background/', file_name)
                    bpy.ops.render.render(write_still=True)
        
        
        
            if os.path.exists(result_path + "/Mask/Image" + format(bpy.context.scene.frame_start + i,"04")+".png"):         
                os.remove(result_path + "/Mask/Image" + format(bpy.context.scene.frame_start + i,"04")+".png")                    
            if os.path.exists(result_path + "/Diffuse/Image" + format(bpy.context.scene.frame_start + i,"04")+".png"):         
                os.remove(result_path + "/Diffuse/Image" + format(bpy.context.scene.frame_start + i,"04")+".png")     
                
                    
                        
                    
    

        
        return{'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


    
def register():
    bpy.utils.register_class(DataSaverMainPanel)
    bpy.utils.register_class(DataSaveSetting_PT_CONTROL)    

    
    
    
def unregister():
    bpy.utils.unregister_class(DataSaverMainPanel)
    bpy.utils.unregister_class(DataSaveSetting_PT_CONTROL)


if __name__ == "__main__":
    register()
    


    
    
    


