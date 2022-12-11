import json
import sys
import os
import cv2
import numpy as np
from mmcv import imshow_bboxes
from pycocotools import mask

#-------------------------------------------------
def rotate_coco(image, angle):
    img = image.copy()
    # init. height and width
    h, w = img.shape[:2]

    # Calculate image size after rotation
    r = np.radians(angle)
    si = np.abs(np.sin(r))
    co = np.abs(np.cos(r))

    nw = int(co*w + si*h)
    nh = int(si*w + co*h)

    # Calculate rotation matrix + add correction due to size change
    center = (w/2, h/2)
    rot_m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_m[0][2] = rot_m[0][2] + (nw - w) // 2
    rot_m[1][2] = rot_m[1][2] + (nh - h) // 2

    # affine transformation
    img = cv2.warpAffine(img, rot_m, (nw, nh), flags=cv2.INTER_CUBIC,borderValue=(128, 128, 128))

    return img, rot_m

#-------------------------------------------------
def main_gen_cocorot(out_dir, json_name_in, json_name_out, root_dir, img_path_dbg, fname_angle):
    
    js_op = open(json_name_in, 'r')
    js_org = json.load(js_op)
    js_out = js_org.copy()

    id_list_ano   = np.array( [ int( js_org['annotations'][k]['image_id'] ) for k in range(len( js_org['annotations'] )) ] )
    id_list_fname = [ js_org['images'][k]['file_name'] for k in range(len( js_org['images'] )) ] 
    id_list_id    = np.array( [ js_org['images'][k]['id'] for k in range(len( js_org['images'] )) ] )


    ##############################
    # read list of angles
    f = open(fname_angle, 'r')
    data = f.read()
    data_angle = data.split('\n')
    num = 0
    for dt in data_angle:
        data_angle[num] = dt.replace(' ', '').split('\t')
        num +=1
    ##############################

    num_img = 0
    for i in range(len(id_list_id)):
        # angle = np.random.rand()*360

        ##############################
        # Check if filenames match
        angle = float(data_angle[num_img][1])
        fname_angle_tmp = data_angle[num_img][0]
        if id_list_fname[i] != fname_angle_tmp:
            print('error: files do not match')
            sys.exit(1)
        num_img += 1
        ##############################

        id_img = id_list_id[i]
        file = root_dir + id_list_fname[i]
        ids_ano =  np.where(id_list_ano == id_img)
        img = cv2.imread(file)
        FILE_IMG_OUT2 = img_path_dbg +'tmp_rotimg_with_annotation/' + id_list_fname[i]

        img_rotate, rot_m = rotate_coco(img, angle)
        cv2.imwrite(out_dir + id_list_fname[i], img_rotate)
        
        all_list = []
        nums = 0
        for k in range(len(ids_ano[0])):
            nums += 1
            id_ano = ids_ano[0][k]
            anodata = js_org['annotations'][id_ano]
            sgm  = anodata['segmentation']

            if 'counts' in sgm:
                compressed_rle = mask.frPyObjects(sgm, sgm.get('size')[0], sgm.get('size')[1])
                msk = mask.decode(compressed_rle)
                sgm_tmp = np.argwhere(msk)
            else:            
                if len(sgm)==1:
                    sgm_tmp = np.array(sgm[0]).reshape(-1,2)
                else:
                    for i in range(len(sgm)):
                        if i==0:
                            sgm_tmp = sgm[0].copy()
                        else:
                            sgm_tmp = sgm_tmp + sgm[i]
                    sgm_tmp = np.array(sgm[0]).reshape(-1,2)
            boxes = []
            for k in sgm_tmp:                
                boxes.append([k[0],k[1],1])
            coord_arr = np.array(boxes)

            # Coordinate transformation by multiplying with rotation matrix
            new_coord = rot_m.dot(coord_arr.T)
            x_ls = new_coord[0]  # x coord,
            y_ls = new_coord[1]  # y  coord,

            # Compute new corner positions using minimum and maximum values
            x = int(min(x_ls))
            y = int(min(y_ls))
            w = int(max(x_ls) - x)
            h = int(max(y_ls) - y)

            js_out['annotations'][id_ano]["bbox"] = [float(x), float(y), float(w), float(h)]
            all_list.append([float(x), float(y), float(x)+float(w), float(y)+float(h)])   
            sgm_tmp = np.concatenate([[x_ls], [y_ls]]).transpose().reshape(-1)
            js_out['annotations'][id_ano]["segmentation"]=[sgm_tmp.tolist().copy()]

        # cv2.imwrite('tmp_bbx_bbx.png',img_rotate)
        print('{} \t{} \t  {}'.format(file, angle, nums))
        with open(img_path_dbg+'rotation_data.txt', mode='a') as f:
            print('{} \t{} \t  {}'.format(file, angle, nums), file=f)

        
        imshow_bboxes(img_rotate, np.array(all_list), out_file=FILE_IMG_OUT2, show=False)
    # write
    fw = open(json_name_out,'w')
    json.dump(js_out,fw,indent=2)


#-------------------------------------------------
if __name__=='__main__':
    
    json_name_in = './coco/instances_val2017.json' 
    json_name_out = './coco_rot/instances_val2017_R.json'
    img_path = './coco_rot/val2017/'
    root_dir= './coco/val2017/'
    img_path_dbg = './coco_rot/val2017_dbg/'
    fname_angle = './accv_val17.txt'
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(img_path_dbg): os.makedirs(img_path_dbg)
    main_gen_cocorot(img_path, json_name_in, json_name_out, root_dir, img_path_dbg, fname_angle)

    json_name_in = './coco/instances_train2017.json' 
    json_name_out = './coco_rot/instances_train2017_R.json'
    img_path = './coco_rot/train2017/'
    root_dir= './coco/train2017/'
    img_path_dbg = './coco_rot/train2017_dbg/'
    fname_angle = './accv_train17.txt'
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(img_path_dbg): os.makedirs(img_path_dbg)
    main_gen_cocorot(img_path, json_name_in, json_name_out, root_dir, img_path_dbg, fname_angle)
