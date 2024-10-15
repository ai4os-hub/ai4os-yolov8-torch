# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:15:47 2024

@author: alebeaud
"""

import os, json
import pandas as pd
import numpy as np
import sys
import warnings
import math
from datetime import datetime
import shutil
import glob as glob
from itertools import combinations
import yaml as yam
import argparse
from ultralytics import YOLO
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Function showing progress on some other functions
def print_progress_lines(progress):
    sys.stdout.write('\rConverting lines, progress: {}%'.format(progress))
    sys.stdout.flush()

def print_progress_points(progress):
    sys.stdout.write('\rConverting points, progress: {}%'.format(progress))
    sys.stdout.flush()

def print_progress_polygons(progress):
    sys.stdout.write('\rConverting polygons, progress: {}%'.format(progress))
    sys.stdout.flush()
    
def print_progress_unite(progress):
    sys.stdout.write('\rCalculating IoUs, progress: {}%'.format(progress))
    sys.stdout.flush()

def print_groups(progress):
    sys.stdout.write('\rOrganizing df, progress: {}%'.format(progress))
    sys.stdout.flush()

def clear_line():
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()

def SaveCSV(df,path,name):
    filename = path + '/' + name + '.csv'
    if os.path.isfile(filename) is True:
        response = input("File already exists, overwrite ? (y/n): ")
        if response.lower() != 'y':
            return
    df.to_csv(filename,sep=',')
    return

# Listing species by alphabetical order. Assigning to each specie a number, useful for later.
def liste_especes (df): 
    # Obtaining the species names in our dataset
    especes=sorted(df['name_sp'].unique())
    # labels = dictionary of numbers associated to each species
    labels={}
    for i, espece in enumerate(especes):
        labels[espece]=i
    return(labels)

# Establishes if 2 angles are close by z degrees
def proche(angle1,angle2,z):
    return abs(angle1 - angle2) < z

# Calculates ratios on the x and y axis
# Those ratios are corrections for each bounding box (converted from lines) that have a close horizontal or vertical angle.
def ratioing(angle_def,length,z):
    ratio_x=0
    ratio_y=0
    if proche(angle_def, 90, z):
        ratio_x = length / 2
    elif proche(angle_def, 180, z) or proche(angle_def,0,z):
        ratio_y = length / 2
    return(ratio_x,ratio_y)

# This function is used for gathering the coordinates for lignesbb
def coord_lignes(w,h,x1,x2,y1,y2):
    # For this, we calculate the euclidian distance of the line
    # We take the absolute values of its coordinates
    x1=abs(x1)
    x2=abs(x2)
    y1=abs(y1)
    y2=abs(y2)
    
    dx = x2 - x1
    dy = y2 - y1
    length=np.sqrt((x2 - x1)**2 + (y2- y1)**2).round(0)

    # Calculate the angle with the x axis
    angle_radians=math.atan2(dy,dx)
    # Conversion of the angles in radians to degrees
    angle = abs(math.degrees(angle_radians))
    
    z = 18

    # Check if the angle calculated is near 90 or 180 degrees +-z° 
    ratios=ratioing(angle,length,z)
    
    ratio_x=ratios[0]
    ratio_y=ratios[1]
    
    X1=min(x1,x2)
    X2=max(x1,x2)
    Y1=min(y1,y2)
    Y2=max(y1,y2)
    
    x_min = max(0,(X1-ratio_x))
    x_max = min(w,(X2+ratio_x))
    y_min = max(0,(Y1-ratio_y))
    y_max = min(h,(Y2+ratio_y))
    
    return(x_min,x_max,y_min,y_max,length)


# Converts lines into bounding boxes
def lines2bb (lignes):
    startTime=datetime.now()
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Copy the starting dataframe (to not overwrite it)
    pls=lignes.copy() 
    pls['type']=str('line')
    
    # Reset the index
    pls=pls.reset_index(drop=True) 
    
    # Image resolution
    w=1920
    h=1080
    
    # Listing final coordinates
    xmin=[]
    xmax=[]
    ymin=[]
    ymax=[]
    length_list=[]
    
    # Total for the datetime function
    total=len(pls.index)
    
    # For every line in our dataframe
    for i in pls.index:
        print_progress_lines(i*100//total)
        
        # You can add a padding to the coordinates before this function
        # Calculates coordinates
        x_min,x_max,y_min,y_max,length=coord_lignes(w,h,pls.iloc[i]['x1'],pls.iloc[i]['x2'],pls.iloc[i]['y1'],pls.iloc[i]['y2'])
        
        # Adding coordinates to the final lists
        xmin.append(x_min)
        xmax.append(x_max)
        ymin.append(y_min)
        ymax.append(y_max)
        length_list.append(pls['length'].iloc[i])
        
    # Adding coordinates to the returned dataframe
    pls['xmin']=xmin
    pls['xmax']=xmax
    pls['ymin']=ymin
    pls['ymax']=ymax
    pls['length']=length_list
    clear_line()
    duration=datetime.now()-startTime
    print('Converted lines, calculation time : {}'.format(str(duration).split('.', 2)[0]))

    return(pls)

# Converts polygons into bounding boxes
def polygones2bb(polygones):
    startTime=datetime.now()
    pls=polygones.copy()
    pls=pls.reset_index(drop=True)
    w,h=1920,1080
    pls['type']=str('polygone')
    
    # Listing final coordinates
    xmin=[]
    xmax=[]
    ymin=[]
    ymax=[]
    
    # Total of lines to show progress
    total=len(pls.index)
    
    for i in pls.index:
        print_progress_polygons(i*100//total)
        
        # Corrects DEEPSEASPY format data
        # Get the coordinates of the polygon
        coords_str=pls["polygon_values"].iloc[i].replace("\\",'') 
        new_coords=coords_str[:2] + '"' + coords_str[2:-1]
        
        # Changes the format of the coordinates
        coords=json.loads(new_coords)

        # Isolates max and min coordinates
        x_min = max(min(coord["x"] for coord in coords),0)
        x_max = min(max(coord["x"] for coord in coords),w)
        y_min = max(min(coord["y"] for coord in coords),0)
        y_max = min(max(coord["y"] for coord in coords),h)
        
        # Adding coordinates to the final lists
        xmin.append(x_min)
        xmax.append(x_max)
        ymin.append(y_min)
        ymax.append(y_max)
    
    # Adding coordinates to the final dataframe
    pls['xmin']=xmin
    pls['xmax']=xmax
    pls['ymin']=ymin
    pls['ymax']=ymax
    clear_line()
    duration = datetime.now() - startTime
    print('Converted polygons, calculation time : {}'.format(str(duration).split('.', 2)[0]))
    return(pls)

# Converts points to bounding boxes
def points2bb(df,buffer=1):
    startTime=datetime.now()
    if type(buffer) != int :
        print('Buffer is not an integer')
        return
    
    pls=df.copy()
    pls=pls.reset_index(drop=True)
    
    pls=pls.dropna(subset=['name_sp'])
    
    pls['type']=str('point')
    
    xmin=[]
    xmax=[]
    ymin=[]
    ymax=[]
    # Dimensions de l'image
    w= 1920
    h= 1080
    total=len(pls.index)
    
    for i in range(len(pls)):
        print_progress_points(i*100//total)
        
        x_min = max(0,(pls['x1'].iloc[i]-buffer))
        x_max = min(w,(pls['x1'].iloc[i]+buffer))
        y_min = max(0,(pls['y1'].iloc[i]-buffer))
        y_max = min(h,(pls['y1'].iloc[i]+buffer))
            
        xmin.append(x_min)
        xmax.append(x_max)
        ymin.append(y_min)
        ymax.append(y_max)
    
    pls['xmin']=xmin
    pls['xmax']=xmax
    pls['ymin']=ymin
    pls['ymax']=ymax
    clear_line()
    duration=datetime.now() - startTime
    print('Converted points, calculation time : {}'.format(str(duration).split('.', 2)[0]))
    return(pls)

# Creates path folder, if ealready exists, erases the folder then recreates it
def stomp(path):
    try :
        os.makedirs(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)

def prepare_yolo (df,path_save,path_img,prop=[.8,.1],method='copy',empty_images=False):
    print('Preparing dataset at location '+ path_save)
    # Converts coordinates in yolo format (from xmin, xmax, ymin, ymax to x, y, w, h)
    # Size [w,h] should contain the width and height of your image(s)
    def convert_yolo(tt,size):
        x=[]
        y=[]
        w=[]
        h=[] 
        dw = 1./size[0]
        dh = 1./size[1]
        for i in range(len(tt)):
            # Finding x and y, the center of your bounding box
            xi = (tt['xmin'].iloc[i] + tt['xmax'].iloc[i])/2.0
            yi = (tt['ymin'].iloc[i] + tt['ymax'].iloc[i])/2.0
            wi = tt['xmax'].iloc[i] - tt['xmin'].iloc[i]
            hi = tt['ymax'].iloc[i] - tt['ymin'].iloc[i]
            
            # Multiply the coordinates by 1/size to scale them between 0 and 1 (yolo format)
            x.append(xi*dw)
            w.append(wi*dw)
            y.append(yi*dh)
            h.append(hi*dh)
        
        tt['x']=x
        tt['y']=y
        tt['w']=w
        tt['h']=h

        return(tt)
    
    # Writes txt files from yolo coordinates
    def encode_yolo(df,path_save,names):
        for i in range(len(names)):
            ff=df[df['name_img']==names[i]]
            name=names[i].replace('.jpg','.txt')
            with open(os.path.join(path_save,(name)),"w",encoding='utf-8') as e:
                for j in range(len(ff)):
                    label = ff['label'].iloc[j]
                    x = round(ff['x'].iloc[j],7)
                    y = round(ff['y'].iloc[j],7)
                    w = round(ff['w'].iloc[j],7)
                    h = round(ff['h'].iloc[j],7)
                    e.write(f"{label} {x} {y} {w} {h}\n")
                
                e.close()
        return()
    
    # Creation of paths and directories for Yolo data repository
    startTime=datetime.now()
    os.makedirs(path_save, exist_ok=True)
    if empty_images==True:
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp'] 
        image_names = []
        for extension in image_extensions:
            image_names.extend(glob.glob(os.path.join(path_img, extension)))
        image_names = [os.path.basename(image) for image in image_names]
    startTime = datetime.now()
    sup_files=['images','labels']
    yolo_images=os.path.join(path_save,(sup_files[0]))
    yolo_labels=os.path.join(path_save,(sup_files[1]))
    
    stomp(yolo_images)
    stomp(yolo_labels)
    
    # Splits data in train, val, test
    sub_df=np.split(df.sample(frac=1), [int(prop[0]*len(df)), int(sum(prop)*len(df))])
    
    #For each folder (train, val set)
    files=['train','val','test']
    for i in range(len(sub_df)):
        # Create the folder for the images and labels
        stomp(os.path.join(yolo_images,files[i]))
        stomp(os.path.join(yolo_labels,files[i]))
        names=sorted(sub_df[i]['name_img'].unique())
        # Write
        file_txt=os.path.join(yolo_labels,(files[i]))
        encode_yolo(sub_df[i],file_txt,names)
        base=os.path.join(yolo_images,files[i])
        for f in names:
            copy_img=os.path.join(path_img,f)
            shutil.copy(copy_img,base)
            # You can change the method by which images are put in the training folder
            # By default, it is by copying so you don't mistakenly delete your data
            # if method=='copy':
            #     shutil.copy(path_img,path_save)
            # elif method=='move':
            #     shutil.move(path_img,path_save)

    print(f"Path :{path_save}")
    duration=datetime.now()-startTime
    print('Finished dataset at location '+path_save+', calculation time : {}'.format(str(duration).split('.', 2)[0]))

###Unification of bounding boxes
# If bounding boxes are overlapping, then the function tries to put them in the same 'group'
# update_groups updates the groups of bb that are being processed
def update_groups(group_dict, index1, index2):
    
    group1 = None
    group2 = None
      
    for group_id, indices in group_dict.items():
        if index1 in indices:
            group1 = group_id
        if index2 in indices:
            group2 = group_id
    
    # If both are already in respective groups, don't unify them
    if group1 is not None and group2 is not None and group1 == group2:
        return group_dict
    
    # If one is in a group and the other is not, add it to the first one's group
    if group1 is not None:
        group_dict[group1].append(index2)
    elif group2 is not None:
        group_dict[group2].append(index1)
    else:
        # If none of them are in a group, create a new group with both of them in it
        new_group_id = max(group_dict.keys(), default=0) + 1
        group_dict[new_group_id] = [index1, index2]
    
    return group_dict

# Tests if bounding boxes are superposed
def superp(bb1,bb2):
    xmin1, xmax1, ymin1, ymax1=bb1[0],bb1[1],bb1[2],bb1[3]
    xmin2, xmax2, ymin2, ymax2=bb2[0],bb2[1],bb2[2],bb2[3]
    if (xmin2 < xmin1 < xmax2 and ((ymin1 < ymin2 < ymax1) or (ymin2 < ymin1 < ymax2))) or (xmin1 < xmin2 < xmax1 and ((ymin1 < ymin2 < ymax1) or (ymin2 < ymin1 < ymax2))):
        return True

# Tests if two bounding boxes are overlapping, then calculates iou
# Basically same as above
def superp_iou(bb1,bb2,s_err=None,nms=False):
    # Gathering coordinates
    xmin1, xmax1, ymin1, ymax1=bb1[0],bb1[1],bb1[2],bb1[3]
    xmin2, xmax2, ymin2, ymax2=bb2[0],bb2[1],bb2[2],bb2[3]

    # Testing if they are overlapping
    if (xmin2 < xmin1 < xmax2 and ((ymin1 < ymin2 < ymax1) or (ymin2 < ymin1 < ymax2))) or (xmin1 < xmin2 < xmax1 and ((ymin1 < ymin2 < ymax1) or (ymin2 < ymin1 < ymax2))):

        #Testing if bb2 is encapsulated in bb1
        if xmin1<xmin2<xmax2<xmax1 and ymin1<ymin2<ymax2<ymax1 :
            return True
        
        #Testing if bb1 is encapsulated in bb2
        if xmin2<xmin1<xmax1<xmax2 and ymin2<ymin1<ymax1<ymax2 :
            return True

        # Sorting coordinates, finding corners
        x_left=max(xmin1,xmin2) #V
        x_right=max(0,min(xmax1,xmax2))
        y_bottom=max(ymin1,ymin2)
        y_top=max(0,min(ymax1,ymax2))
        
        #inter_box=[x_left,x_right,y_bottom,y_top]
        
        # Calculus of the intersection area between the two bb
        intersection_area = (x_right - x_left) * (y_top - y_bottom)
        
        # Calculus of the area of the 2 bb
        bb1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
        bb2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
        if nms==True:
            iou = intersection_area / min(bb2_area,bb1_area)
        else:
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        if iou == 0:
            return(None)
        if iou >= s_err:  # Adjust the IoU threshold as needed
            return True

# Unification of bounding boxes
def unite(df_og, iou_thresh=None, grouper_0=False, nms=False):
    
    # Shows how long the process is at the end
    startTime = datetime.now()
    # Avoids certain warnings 
    pd.options.mode.chained_assignment = None
    # Copy the original df
    df = df_og.copy()
    
    # Setting some default parameters
    # Mets les groupes à 0 pour toutes les lignes
    df['group'] = 0
    # Dictionary for every group
    group_dict={}
    
    # Gets every image name
    images = df['name_img'].unique()
    total = len(images)
    
    # Adds label column, which associates to every species a specific number
    labels=liste_especes(df)
    df['label']=df['name_sp'].map(labels)
    
    # For every image
    for i, image in enumerate(images):
        
        # Subset of orginal df with only image i
        tt_i = df[df['name_img'] == image]
        labels_i = tt_i['name_sp'].unique()
        print_progress_unite(i*100//total)
        
        # For each species in the image
        for label in labels_i:
            tt_il = tt_i[tt_i['name_sp'] == label]
            index_il = tt_il.index
            
            # For every bb of the species
            for ii, jj in combinations(index_il, 2):
                # Getting 2 bb
                bb1 = [tt_i['xmin'][ii], tt_i['xmax'][ii], tt_i['ymin'][ii], tt_i['ymax'][ii]]
                bb2 = [tt_i['xmin'][jj], tt_i['xmax'][jj], tt_i['ymin'][jj], tt_i['ymax'][jj]]
                
                # If there is no threshold, only checks if the bounding boxes are overlapping
                if iou_thresh is None :
                    # Compares 2 bb 
                    if superp(bb1,bb2)==True:
                        update_groups(group_dict, ii, jj)
                        
                else :
                    # Compares 2 bb with a specific IoU threshold
                    if superp_iou(bb1,bb2,iou_thresh,nms)==True:
                        update_groups(group_dict, ii, jj)

    # Update dictionary based on the index
    for group_id, indices in group_dict.items():
        df.loc[indices, 'group'] = group_id
    
    # Sorting groups
    groupes=sorted(df['group'].unique())
    if grouper_0==True:
        group_0=df[df['group']==0]
        group_0=group_0[['name_img','name_sp','label','xmin', 'xmax', 'ymin', 'ymax','group','length']]
    # Removing line 0 for groups to avoid looping problems
    #groupes.remove(0)
    
    # Composition of the final df
    # Columns :
    result = pd.DataFrame(columns=['name_img','name_sp','label','xmin', 'xmax', 'ymin', 'ymax','group','length','occurences'])
    # For each group
    clear_line()
    for group in groupes:
        print_groups(group*100//len(groupes))
        temp=df[df['group'] == group]
        
        row=temp.iloc[0]
        
        # Each row is composed of :
        row_data = {
              'name_img': row['name_img'],
              'name_sp': row['name_sp'],
              'label': row['label'],
              'xmin': temp['xmin'].min(),
              'xmax': temp['xmax'].max(),
              'ymin': temp['ymin'].min(),
              'ymax': temp['ymax'].max(),
              'length': row['length'],
              'group': int(group),
              'occurences': len(temp)
        }
        # Converts the row in df format
        dfd_data=pd.DataFrame([row_data])
        # Appends the row to the final df
        result = pd.concat([result, dfd_data], ignore_index=True)
    
    
    #df=result[result['group'] != 0]
    if grouper_0==True:
        group_0['occurences']=None
        df=pd.concat([result,group_0],axis=0,join='outer')
        df=df[1:]
        df.reset_index(inplace=True) 
    else :
        # Excluding group 0
        df=result[result['group'] != 0]
    clear_line()
    duration=datetime.now()-startTime
    print('Finished the unification of bounding boxes, calculation time : {}'.format(str(duration).split('.', 2)[0]))
    return df

# Creates a yaml file with information regarding your data paths for Yolov8
def create_yaml(df,path_save, output='output'):
    startTime = datetime.now()
    train_path = f"{path_save}/images/train"
    val_path = f"{path_save}/images/val"
    test_path = f"{path_save}/images/test"

    
    
    # Creates a dictionary of all species
    names_dict = df.set_index('label')['name_sp'].to_dict()
    
    # Naming the future .yaml file
    
    output_file=f"{output}.yaml"
    # Elements of the future .yaml file
    yaml_content = {
        'path': path_save,
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'names': names_dict
    }
    
    # Getting main working directory
    home=os.getcwd()
    
    # Changing working directory for the storing of the .yaml file
    if os.path.exists(path_save)==True:
        os.chdir(path_save)
    else:
        os.makedir(path_save)
        os.chdir(path_save)
    
    print('Creating yaml file...')
    # Creating the .yaml file
    with open(output_file, 'w') as file:
        dump = yam.dump(yaml_content, default_flow_style = False, allow_unicode = True, encoding = None, sort_keys=False)
        file.write( dump )
        
    # Return to main working directory
    os.chdir(home)
    duration=datetime.now()-startTime
    sys.stdout.write('\r' + ' ' * len('Creating yaml file...') + '\r')
    print('Created .yaml file at the location '+path_save+', calculation time : {}'.format(str(duration).split('.', 2)[0]))
    print(yaml_content)
    
def encode_yolo(df,path_save,names):
    for i in range(len(names)):
        ff=df[df['name_img']==names[i]]
        name=names[i].replace('.jpg','.txt')
        with open(os.path.join(path_save,(name)),"w",encoding='utf-8') as e:
            for j in range(len(ff)):
                label = ff['label'].iloc[j]
                x = round(ff['x'].iloc[j],7)
                y = round(ff['y'].iloc[j],7)
                w = round(ff['w'].iloc[j],7)
                h = round(ff['h'].iloc[j],7)
                e.write(f"{label} {x} {y} {w} {h}\n")
            
            e.close()
    return()

def convert_yolo(tt,size):
    x=[]
    y=[]
    w=[]
    h=[] 
    dw = 1./size[0]
    dh = 1./size[1]
    for i in range(len(tt)):
        # Finding x and y, the center of your bounding box
        xi = (tt['xmin'].iloc[i] + tt['xmax'].iloc[i])/2.0
        yi = (tt['ymin'].iloc[i] + tt['ymax'].iloc[i])/2.0
        wi = tt['xmax'].iloc[i] - tt['xmin'].iloc[i]
        hi = tt['ymax'].iloc[i] - tt['ymin'].iloc[i]
        
        # Multiply the coordinates by 1/size to scale them between 0 and 1 (yolo format)
        x.append(xi*dw)
        w.append(wi*dw)
        y.append(yi*dh)
        h.append(hi*dh)
    
    tt['x']=x
    tt['y']=y
    tt['w']=w
    tt['h']=h

    return(tt)

def parse_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):  # Ignore empty lines and comments
                key, value = line.strip().split('=')
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.replace('.', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
                config[key] = value
    return config


def reset_label(df):
    labels=liste_especes(df)
    df['label']=df['name_sp'].map(labels)
    return(df)

#def run_script(path_imgs=None, path_csv=None, path_save=None, polygons=False, points=False, lines=False, iou=0.2):
def run_script(path_imgs, path_csv, path_save, polygons, points, lines, iou,
               model_path, epochs, imgsz, batch_size, device, workers, patience, save, save_period, cache, project, name,
               exist_ok, pretrained, optimizer, verbose, seed, deterministic, single_cls, rect, close_mosaic, resume, amp,
               fraction, profile, freeze, lr0, lrf, momentum, weight_decay, warmup_epochs, warmup_momentum,
               warmup_bias_lr, box, cls, dfl, pose, kobj, label_smoothing, nbs, overlap_mask, mask_ratio, dropout, val, plots):
    path_yaml=os.path.join(path_save,'output.yaml')
    if os.path.exists(path_yaml):
        response = input('found output.yaml file, do you want to skip the cleaning and train yolov8 ? y/n')
        if response.lower() == 'y':
            yaml_path=os.path.join(path_save,output)
            
            print('Launching the training of YoloV8')
            model = YOLO('yolov8n.pt')

            model.train(
            data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch_size, device=device, workers=workers, patience=patience,
            save=save, save_period=save_period, cache=cache, project=project, name=name, exist_ok=exist_ok,
            pretrained=pretrained, optimizer=optimizer, verbose=verbose, seed=seed, deterministic=deterministic,
            single_cls=single_cls, rect=rect, close_mosaic=close_mosaic, resume=resume, amp=amp, fraction=fraction,
            profile=profile, freeze=freeze, lr0=lr0, lrf=lrf, momentum=momentum, weight_decay=weight_decay,
            warmup_epochs=warmup_epochs, warmup_momentum=warmup_momentum, warmup_bias_lr=warmup_bias_lr, box=box,
            cls=cls, dfl=dfl, pose=pose, kobj=kobj, label_smoothing=label_smoothing, nbs=nbs, overlap_mask=overlap_mask,
            mask_ratio=mask_ratio, dropout=dropout, val=val, plots=plots)
            print('Training completed')
        else:
            print('Cleaning the dataset.')

    path_img=Path(path_imgs)
    polybb, lignesbb, pointsbb = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    data=pd.read_csv(path_csv,sep=None,engine='python',on_bad_lines='skip')
    # Rename your columns so that our functions run properly
    #nécessaire sur du deepseaspy
    # data.rename(columns={'pos1x': 'x1', 'pos1y': 'y1','pos2x': 'x2', 'pos2y': 'y2','name_fr':'name_sp','name':'name_img'}, inplace=True)
    # Subset of polygon labels
    if polygons == True:
        poly=data.dropna(subset=['polygon_values'])
        polybb=polygones2bb(poly)
    # Subset of points labels
    if points == True:
        points=data[data['polygon_values'].isna() & data['x2'].isna()]
        pointsbb=points2bb(points)    
    # Subset of lines labels
    if lines == True:
        lines=data.dropna(subset=['x2'])
        lignesbb=lines2bb(lines)
    bb=pd.concat([polybb,lignesbb,pointsbb])
    if lines.empty and polygons.empty and points.empty :
        bb=data
    # Save the converted dataset
    SaveCSV(bb,path_save,'export_bb')
    tbc=unite(bb,iou)
    ubb=unite(tbc,0.5,nms=True)
    ubb_rs=reset_label(ubb)
    # Save the converted dataset
    SaveCSV(ubb_rs,path_save,'export_ubb')
    #Size of your images
    size=[1920,1080]

    convert_yolo(ubb_rs,size)
    prepare_yolo(ubb_rs,path_save,path_img,prop=[.8,.1]) 
    
    # Creates a yaml file containing all of the information necessary for running Yolov8 on your data
    create_yaml(ubb_rs,path_save,'output')
    
    # Get where the .yaml file is stored
    output=str('output'+'.yaml')
    yaml_path=os.path.join(path_save,output)
    
    print('Launching the training of YoloV8')
    model = YOLO('yolov8n.pt')

    model.train(
    data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch_size, device=device, workers=workers, patience=patience,
    save=save, save_period=save_period, cache=cache, project=project, name=name, exist_ok=exist_ok,
    pretrained=pretrained, optimizer=optimizer, verbose=verbose, seed=seed, deterministic=deterministic,
    single_cls=single_cls, rect=rect, close_mosaic=close_mosaic, resume=resume, amp=amp, fraction=fraction,
    profile=profile, freeze=freeze, lr0=lr0, lrf=lrf, momentum=momentum, weight_decay=weight_decay,
    warmup_epochs=warmup_epochs, warmup_momentum=warmup_momentum, warmup_bias_lr=warmup_bias_lr, box=box,
    cls=cls, dfl=dfl, pose=pose, kobj=kobj, label_smoothing=label_smoothing, nbs=nbs, overlap_mask=overlap_mask,
    mask_ratio=mask_ratio, dropout=dropout, val=val, plots=plots)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Citizen science data cleaning & yolo detection.')
    parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
    args = parser.parse_args()

    config = parse_config(args.config)

run_script(
        path_imgs=config['path_imgs'],
        path_csv=config['path_csv'],
        path_save=config['path_save'],
        polygons=config.get('polygons', False),
        points=config.get('points', False),
        lines=config.get('lines', False),
        iou=config.get('iou', 0.2),
        model_path=config.get('model', 'yolov8n.pt'),
        epochs=config.get('epochs', 10),
        imgsz=config.get('imgsz', 640),
        batch_size=config.get('batch', 16),
        device=config.get('device', None),
        workers=config.get('workers', 8),
        patience=config.get('patience', 100),
        save=config.get('save', True),
        save_period=int(config.get('save_period', -1)),
        cache=config.get('cache', False),
        project=config.get('project', None),
        name=config.get('name', None),
        exist_ok=config.get('exist_ok', False),
        pretrained=config.get('pretrained', True),
        optimizer=config.get('optimizer', 'auto'),
        verbose=config.get('verbose', False),
        seed=config.get('seed', 0),
        deterministic=config.get('deterministic', True),
        single_cls=config.get('single_cls', False),
        rect=config.get('rect', False),
        close_mosaic=config.get('close_mosaic', 10),
        resume=config.get('resume', False),
        amp=config.get('amp', False),
        fraction=config.get('fraction', 1.0),
        profile=config.get('profile', False),
        freeze=config.get('freeze', None),
        lr0=config.get('lr0', 0.01),
        lrf=config.get('lrf', 0.01),
        momentum=config.get('momentum', 0.937),
        weight_decay=config.get('weight_decay', 0.0005),
        warmup_epochs=config.get('warmup_epochs', 3.0),
        warmup_momentum=config.get('warmup_momentum', 0.8),
        warmup_bias_lr=config.get('warmup_bias_lr', 0.1),
        box=config.get('box', 7.5),
        cls=config.get('cls', 0.5),
        dfl=config.get('dfl', 1.5),
        pose=config.get('pose', 12.0),
        kobj=config.get('kobj', 2.0),
        label_smoothing=config.get('label_smoothing', 0.0),
        nbs=config.get('nbs', 64),
        overlap_mask=config.get('overlap_mask', True),
        mask_ratio=config.get('mask_ratio', 4),
        dropout=config.get('dropout', 0.0),
        val=config.get('val', True),
        plots=config.get('plots', False)
    )

