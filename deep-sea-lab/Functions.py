# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:45:43 2023

@author: alebeaud
"""

#Import necessary packages
from pathlib import Path
import os, json, random
import pandas as pd
import numpy as np
from PIL import Image
import sys 
import cv2
import warnings
import math
from datetime import datetime
import matplotlib.pyplot as plt
import shutil
import glob as glob
from itertools import combinations
import yaml as yam
import torch
import torchvision.ops.boxes as bops

# Suppress some FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Function showing progress on some other functions
def print_progress(progress):
    sys.stdout.write('\rProgress: {}%'.format(progress))
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

# Listing species by alphabetcal order. Assigning to each specie a number, useful for later.
def liste_especes (df): #Add sortie df
    # Obtaining the species names in our dataset
    especes=sorted(df['name_sp'].unique())
    # labels = dictionary of numbers associated to each species
    labels={}
    for i, espece in enumerate(especes):
        labels[espece]=i
    return(labels)

# Allows you to plot lines in your python visualizer, without drawing on your images (resolution 1920x1080)
def plot_line(x1,y1,x2,y2):
    # Creates fig and subplots
    fig, ax = plt.subplots()
    
    # Draws a line between the two points
    ax.plot([x1, x2], [y1, y2], marker='o', linestyle='-', color='b')
    
    # Adds etiquettes to points
    ax.text(x1, y1, f'({x1}, {y1})', fontsize=12, ha='right')
    ax.text(x2, y2, f'({x2}, {y2})', fontsize=12, ha='right')
    
    # Delimit the axes
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    
    # Plot
    plt.show()

# Allows you to plot bounding boxes in your python visualizer, without drawing on your images (resolution 1920x1080)
def plot_bb(x1,y1,x2,y2):
    #Creates fig and subplots
    fig, ax = plt.subplots()
    
    # Draws the bounding box based on two points
    cv2.rectangle((min(x1,x2),min(y1,y2)),(max(x1,x2),max(y1,y2)), 'red', 2)
    
    # Adjust the X and Y limits
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    
    # Plot 
    plt.show()

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
    print('Converting lines...')
    # For every line in our dataframe
    for i in pls.index:
        progress = (i+1) * 100 // total
        if progress in [25, 50, 75, 100]:
            print_progress(progress)
        
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
    print('Calculation time : {}'.format(str(duration).split('.', 2)[0]))

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
    
    print('Converting polygons...')
    # For every line in our dataframe
    for i in pls.index:
        progress = (i+1) * 100 // total
        if progress in [25, 50, 75, 100]:
            print_progress(progress)
        
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
    print('Calculation time : {}'.format(str(duration).split('.', 2)[0]))
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
    

    print('Converting points...')
    # For every line in our dataframe
    for i in range(len(pls)):
        progress = (i+1) * 100 // total
        if progress in [25, 50, 75, 100]:
            print_progress(progress)
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
    print('Calculation time : {}'.format(str(duration).split('.', 2)[0]))
    return(pls)

# Tests if bounding boxes are superposed
def superp(bb1,bb2):
    xmin1, xmax1, ymin1, ymax1=bb1[0],bb1[1],bb1[2],bb1[3]
    xmin2, xmax2, ymin2, ymax2=bb2[0],bb2[1],bb2[2],bb2[3]
    if (xmin2 < xmin1 < xmax2 and ((ymin1 < ymin2 < ymax1) or (ymin2 < ymin1 < ymax2))) or (xmin1 < xmin2 < xmax1 and ((ymin1 < ymin2 < ymax1) or (ymin2 < ymin1 < ymax2))):
        return True

# Calculates iou between bb1 and bb2
def iou_calc(bb1,bb2,s_err=None):
    # Gathering coordinates
    xmin1, xmax1, ymin1, ymax1=bb1[0],bb1[1],bb1[2],bb1[3]
    xmin2, xmax2, ymin2, ymax2=bb2[0],bb2[1],bb2[2],bb2[3]
    
    # Sorting coordinates, finding corners
    XOI=max(xmin1,xmin2)
    XII=max(0,min(xmax1,xmax2))
    YOI=max(ymin1,ymin2)
    YII=max(0,min(ymax1,ymax2))
    
    # Calculus of the intersection area between the two bb
    interArea = abs(max(abs(XII - XOI), 0) * max(abs(YII - YOI), 0))
    
    # Calculus of the area of the 2 bb
    boxAArea = abs((xmax1 - xmin1) * (ymax1 - ymin1))
    boxBArea = abs((xmax2 - xmin2) * (ymax2 - ymin2))

    # Discards calculus if result is 0
    if float(boxAArea + boxBArea - interArea) == 0:
        return(None)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    if iou >= s_err:  # Add the thresh if needed
        return(iou)


# Converts coordinates in yolo format (from xmin, xmax, ymin, ymax to x, y, w, h)
# Size [w,h] should contain the width and height of your image(s)
def convert_yolo(tt,path_img):
    x=[]
    y=[]
    w=[]
    h=[] 
    #dw = 1./size[0]
    #dh = 1./size[1]
    print('Converting coordinates to Yolo format...')
    for i in range(len(tt)):
        progress = (i+1) * 100 // len(tt)
        if progress in [25, 50, 75, 100]:
            print_progress(progress)
        im = cv2.imread(os.path.join(path_img,tt['name_img'].iloc[i]))
        dw = 1./im.shape[1]
        dh = 1./im.shape[0]
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

def generate_color_dict(animals, colormap_name='viridis'):
    num_animals = len(animals)
    colors = {}

    if colormap_name == 'red':
        for animal in animals:
            colors[animal] = (0, 0, 255)  # Rouge en RGB
    else :
        colormap = plt.get_cmap(colormap_name)
        indices = np.linspace(0, 1, num_animals)  # Divise la colormap en segments égaux
        for i, animal in enumerate(animals):
            rgba_color = colormap(indices[i])
            rgb_color = tuple(int(255 * c) for c in rgba_color[:3])  # Convertir en valeurs RGB 0-255
            colors[animal] = rgb_color
    return colors

# Allows the user to visualize data with bounding boxes
# nb_img asks for a number of images, randomly picked in your df/data
# colors
def vision(df,path_img,path_save=None,nb_img=None,colors=None):
    # Get every name imgs in df
    startTime=datetime.now()
    names=sorted(df['name_img'].unique())
    
    # Determine the directory to save the images
    if path_save is None:
        parent_directory = os.path.dirname(path_img)
        stock_directory = os.path.join(parent_directory, 'stock')
    else:
        stock_directory = os.path.join(path_save, 'stock')
    
    # Check if the save directory exists
    if os.path.exists(stock_directory):
        print("Stock already exists.")
        count = 1
        base_stock_directory = stock_directory  # Save the base path for future use
        # Find a new directory name that does not already exist
        while os.path.exists(stock_directory):
            stock_directory = f"{base_stock_directory}{count}"
            count += 1
        os.makedirs(stock_directory)
        print(f"New 'stock' directory created at {stock_directory}")
    else:
        # If the directory does not exist, create it
        os.makedirs(stock_directory)
        print(f"New 'stock' directory created at {stock_directory}")
        
    if nb_img <= len(df['name_img'].unique()) :
        # Randomly picks images
        nb_index=random.sample(range(len(names)),nb_img)
        index={i: names[i] for i in nb_index}
        # Select the lines corresponding to the random images
        ff = df[df['name_img'].isin(index.values())]
    else :
        ff = df
    
    if colors==None:
        animals=df['name_sp'].unique()
        colors=generate_color_dict(animals,'viridis')
    if colors=='red':
        animals=df['name_sp'].unique()
        colors=generate_color_dict(animals,'red')
    
    names_flt=sorted(ff['name_img'].unique())
    total=len(names_flt)
    for c, i in enumerate(names_flt):
        print_progress(c*100//total)
        # Open the image
        img = cv2.imread(str(os.path.join(path_img,i)))
        img_h, img_w = img.shape[:2]
        filtered_df=ff[ff['name_img']==i]
    
        for j in range(len(filtered_df)):
            # Plot bb
            x_min = int(filtered_df['xmin'].iloc[j])
            y_min = int(filtered_df['ymin'].iloc[j])
            x_max = int(filtered_df['xmax'].iloc[j])
            y_max = int(filtered_df['ymax'].iloc[j])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=colors[filtered_df['name_sp'].iloc[j]], thickness=2)
        
        # Save image
        path_sve=os.path.join(stock_directory,i)
        if os.path.exists(path_sve) :
            # Append a suffix to the file name if it already exists
            name, extension = os.path.splitext(i)
            count = 1
            while os.path.exists(os.path.join(stock_directory, f"{name}_{count}{extension}")):
                count += 1
            path_sve = os.path.join(stock_directory, f"{name}_{count}{extension}")
        cv2.imwrite(os.path.join(stock_directory,i), img)
    clear_line()
    print('Plotting done')
    duration=datetime.now()-startTime
    print('Calculation time : {}'.format(str(duration).split('.', 2)[0]))

# Converts back from yolo format to min/max coordinates
def minmax(row):
    xmin=max(row['X']-(row['Width']/2),0)
    xmax=min(row['X']+(row['Width']/2),1920)
    ymin=max(row['Y']-(row['Height']/2),0)
    ymax=min(row['Y']+(row['Height']/2),1080)
    length=np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2).round(0)
    return(pd.Series({'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'length': length}))

# Apply minmax on all rows
def update_df(df):
    bb = df.copy()
    bb[['xmin', 'xmax', 'ymin', 'ymax','length']] = bb.apply(minmax, axis=1)
    return bb

# Returns the name of images in df but not in path_img
def check_img_df (df, path_img):
    
    pas_de_lbl=[]
    part=[]
    
    listJpeg=list(path_img.glob('**/*.jpg'))
    for i in range(len(listJpeg)):
        part.append(str(listJpeg[i].parts[-1]))

    part = sorted(part)
    names = sorted(df['name_img'].unique())
    
    for i in range(len(names)):
        if names[i] not in part:
            pas_de_lbl.append(names[i])
    if len(pas_de_lbl) !=0 and len(part) != len(names):
        print("The number of images in path_img does not match the nmber of images on the df. Verify your data before proceeding.")
    return(pas_de_lbl)

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

# Creates path folder, if ealready exists, erases the folder then recreates it
def stomp(path):
    try :
        os.makedirs(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)

# Prepare the yolo data repositories
# This function copies images, it can generate a lot of data depending on their size
# Also, you can change the way images are copied by writing method='move' and uncommenting the corresponding rows below
# Because it is risky (since you could erase your data by accident), we commented it
# Still, it may be useful depending on your needs
def prepare_yolo (df,path_save,path_img,prop=[.8,.1],method='copy',empty_images=False):
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
    print('Preparing Yolo training dataset...')
    for i in range(len(sub_df)):
        progress = (i+1) * 100 // len(sub_df)
        if progress in [25, 50, 75, 100]:
            print_progress(progress)
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
    print('Calculation time : {}'.format(str(duration).split('.', 2)[0]))

def prepare_yolo_monofolder (df,path_save,path_img,method='copy',empty_images=False):
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
    #sub_df=np.split(df.sample(frac=1), [int(prop[0]*len(df)), int(sum(prop)*len(df))])
    sub_df=df.copy
    #For each folder (train, val set)
    #files=['train','val','test']
    files=['main_yolo_folder']
        # Create the folder for the images and labels
    stomp(os.path.join(yolo_images,files[0]))
    stomp(os.path.join(yolo_labels,files[0]))
    names=sorted(sub_df[0]['name_img'].unique())
    # Write
    file_txt=os.path.join(yolo_labels,(files[0]))
    encode_yolo(sub_df[0],file_txt,names)
    base=os.path.join(yolo_images,files[0])
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
    print('Calculation time : {}'.format(str(duration).split('.', 2)[0]))


# Catalog allows you to extract bb out of your images to easier verify them 
# Deleting a thubmnail and then using the next function (get_df) will give you a df without deleted bb
def catalog(df, path_img, path_save=None,padding=0):
    startTime=datetime.now()
    
    # Lists images
    list_img=list(path_img.glob('**/*.jpg'))
    
    if path_save==None:
        parent_directory = os.path.dirname(path_img)
        stock_directory = os.path.join(parent_directory, 'catalog')
    else:
        stock_directory= os.path.join(path_save, 'catalog')
    # If path_out doesn't exist, create it
    if not os.path.exists(stock_directory): 
        os.makedirs(stock_directory)
        
    unique_name_sp = df['name_sp'].unique()
    
    # For each specie
    # Create a directory that will contain the species images
    for name in unique_name_sp:
        folder_path = os.path.join(stock_directory, name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else :
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
    
    # For each image
    total=len(list_img)
    print('Cataloguing snapshots...')
    for i, img in enumerate(list_img):
        progress = (i+1) * 100 // total
        if progress in [25, 50, 75, 100]:
            print_progress(progress)
        im = Image.open(img)
        image=img.parts[-1]
        sdf = df[df['name_img'] == image]
        
        # Extract the bb
        for i in range(len(sdf)):
            bb=[sdf['xmin'].iloc[i]-padding,sdf['xmax'].iloc[i]+padding,sdf['ymin'].iloc[i]-padding,sdf['ymax'].iloc[i]+padding]
            im1=im.crop((bb[0],bb[2],bb[1],bb[3]))
            name_img=str(image.rstrip('.jpg')+'.'+str(sdf.index[i])+'.jpg')
            
            # Save the exctracted bb
            output_path = os.path.join(stock_directory, sdf['name_sp'].values[i], name_img)
            try:
                im1.save(output_path)
            except ValueError:
                print("Error while printing file, "+output_path+". The function will continue anyway")
                continue
    clear_line()
    print(datetime.now() - startTime)
    
# Allows you to get a df containing only the remaining thumbnails after catalog
# Only to use if you have deleted bounding boxes
def get_df(df,path_save):
    # Getting the catalog directory
    catalog_path = os.path.join(path_save, 'catalog')
    if os.path.exists(catalog_path)==True:
        path_catal=catalog_path
    elif os.path.basename(path_save) == 'catalog' and os.path.exists(path_save):
        path_catal=path_save
    else :
        raise FileNotFoundError("Can't find the path to catalog, make sure it's present in your save directory and correctly named 'catalog'")
    
    liste_index=[]
    # Gets every species name from directories names
    unique_name_sp = [d for d in os.listdir(path_catal) if os.path.isdir(os.path.join(path_catal, d))]
    # For each species
    for name in unique_name_sp:
        folder_path = Path(os.path.join(path_catal, name))
        # Gather images names
        list_img=list(folder_path.glob('**/*.jpg'))
        for i in range(len(list_img)):
            # Get the image name and 'extension' from the filename,
            #which is used as an index to find the corresponding row
            nom_base, extension = os.path.splitext(list_img[i])
            index = int(nom_base.split('.')[-1])
            liste_index.append(index)
    df_filtre = df.loc[liste_index]
    return(df_filtre)

# Creates a yaml file with information regarding your data paths for Yolov8
def create_yaml(df,path_save, output='output'):
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
    
    # Creating the .yaml file
    with open(output_file, 'w') as file:
        dump = yam.dump(yaml_content, default_flow_style = False, allow_unicode = True, encoding = None, sort_keys=False)
        file.write( dump )
        
    # Return to main working directory
    os.chdir(home)
    print(yaml_content)

# Finds empty images based on your available images and your data
def find_empty_images(df, path_img):
    end = []
    try :
        df.reset_index(inplace=True) 
    except :
        ValueError()
    # Gathering info
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp'] 
    image_names = []
    list_df=df['name_img'].unique().tolist()
    for extension in image_extensions:
        image_names.extend(glob.glob(os.path.join(path_img, extension)))
    # Getting the images names
    image_names = [os.path.basename(image) for image in image_names]
    for i in range(len(image_names)): 
        if image_names[i] not in list_df:
            end.append(image_names[i])
    print("Total number of images")
    print(len(image_names))
    print("Images annotated in the dataframe")
    print(len(list_df))
    print("Difference")
    print(abs(len(image_names)-len(list_df)))
    print("Supposedly empty images")
    print(len(end))
    # Return empty images
    return end

# Deletes the labels that have the same name as the imgs in the path_img
# Can be useful if you want to assure certain images that are supposedly empty to not have any annotations
def reorganize_empty(df,path_img,path_save,trashcan,prop=[.8,.1]): 
    sup_files=['images','labels']
    files=['train','val','test']
    
    yolo_images=os.path.join(path_save,(sup_files[0]))
    yolo_labels=os.path.join(path_save,(sup_files[1]))
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp'] 
    
    image_names = []
    for extension in image_extensions:
        image_names.extend(glob.glob(os.path.join(path_img, extension)))
    image_names = [os.path.basename(image) for image in image_names]
    
    for i in range(len(image_names)): 
        for file in files:
            label=image_names[i].replace('.jpg','.txt')
            path_to_img=os.path.join(yolo_images,file)
            path_to_lbl=os.path.join(yolo_labels,file)
            to_del_img=os.path.join(path_to_img,image_names[i])
            to_del_lbl=os.path.join(path_to_lbl,label)
            try :
                shutil.copy(to_del_img, trashcan)
                os.remove(to_del_img)
                shutil.copy(to_del_lbl, trashcan)
                os.remove(to_del_lbl)
            except (FileNotFoundError,):
                continue
    
    # Randomly organize images
    df_reduit=pd.DataFrame(image_names)
    random_indices = np.random.permutation(len(df_reduit))
    
    # Divide the dataset in train, test, val
    train_size = int(prop[0] * len(df_reduit))
    val_size = int(prop[1] * len(df_reduit))
    
    # Getting indices from the 3 datasets
    train_indices = random_indices[:train_size]
    val_indices = random_indices[train_size:train_size + val_size]
    test_indices = random_indices[train_size + val_size:]
    
    # Exctract data for each datasets, from the indices lists aboves
    sub_lists = [df_reduit.iloc[train_indices], df_reduit.iloc[val_indices], df_reduit.iloc[test_indices]]
    
    # Organize and write data in the yolo format
    files=['train','val','test']
    for i in range(len(sub_lists)):
        print("Empty images :"+files[i]+" size")
        print(len(sub_lists[i]))
        sous_df=sub_lists[i]
        
        sous_df=sous_df.reset_index(drop=True)
        path_images=os.path.join(yolo_images,files[i])
        for j in range(len(sous_df)):
            copy_img=os.path.join(path_img,sous_df.iloc[j,0])
            try :
                shutil.copy(copy_img,path_images)
                #shutil.copy(copy_img,trashcan)
            except (FileNotFoundError,):
                print('fichier '+copy_img+' non trouvé')
                continue

# Limits the dataframe to l images
# Useful for testing on resctricted datasets
def limit(df,l):
    poww=pd.DataFrame()
    for i in df['name_img'].unique():
        loww=df[df['name_img']==i]
        if len(loww)>l:
            poww=pd.concat([poww,loww])
    return poww

# Copies empty images and modifies your labels to match it
# To be used if you are certain that some of your data should stay unlabeled
# Clears labels corresponding to the empty images
# list_empty is a list of empty images that you have set yourself or with find_empty
def copy_empty_images(list_empty,df_origin, path_img, path_save,prop=[.8,.1]):
    
    sup_files=['images']
    files=['train','val','test']
    
    yolo_images=os.path.join(path_save,(sup_files[0]))
    
    stomp(yolo_images)
    
    # 10% of empty images is recommended for Yolo
    # Limit the number of empty images to 10% per subset (train, test, val)
    len_rand = int(len(df_origin)/10) 
    if len(list_empty)<len_rand :
        df_reduit = pd.DataFrame(list_empty)
    else:
        list_rand = list_empty.sample(n=len_rand, random_state=42)
        df_reduit = pd.DataFrame(list_rand)
    
    # Randomly change row indices
    random_indices = np.random.permutation(len(df_reduit))
    
    # Divide rows in 2 different datasets, test is deduced by taking the remaining rows
    train_size = int(prop[0] * len(df_reduit))
    val_size = int(prop[1] * len(df_reduit))
    
    train_indices = random_indices[:train_size]
    val_indices = random_indices[train_size:train_size + val_size]
    test_indices = random_indices[train_size + val_size:]
    
    # Use indices as indexes for the roxq to be exctracted from the main df
    sub_lists = [df_reduit.iloc[train_indices], df_reduit.iloc[val_indices], df_reduit.iloc[test_indices]]
    
    files=['train','val','test']
    for i in range(len(sub_lists)):
        print("Empty images :"+files[i]+" size")
        print(len(sub_lists[i]))
        stomp(os.path.join(yolo_images,files[i]))
        path_images=os.path.join(yolo_images,files[i])
        for image in sub_lists[i] :
            copy_img=os.path.join(path_img,image)
            shutil.copy(copy_img,path_images)

# Changes the yolo validation labels by data from another dataframe
# Useful if you have validating data and you want to change test and val
# Images are not modified, only labels
def change_val(df_import,path_dataset):
    # Preparation writes labels from a df, and can overwrite one already existing at this location
    def preparation(df,path_labels):
        stomp(path_labels)
        names=sorted(df['name_img'].unique())
        encode_yolo(df,path_labels,names)

    sup_files=['images','labels']
    low_files=['test','val']
    
    # Gathering paths
    yolo_images=os.path.join(path_dataset,(sup_files[0]))
    yolo_labels=os.path.join(path_dataset,(sup_files[1]))
    images_test=Path(os.path.join(yolo_images,(low_files[0])))
    images_val=Path(os.path.join(yolo_images,(low_files[1])))
    labels_test=os.path.join(yolo_labels,(low_files[0]))
    labels_val=os.path.join(yolo_labels,(low_files[1]))
    
    # Getting images names from test and val folders
    list_test=list(images_test.glob('**/*.jpg'))
    list_val=list(images_val.glob('**/*.jpg'))
    
    
    file_names_test = list(map(os.path.basename, list_test))
    # Filter the df based on the retained images
    file_test = pd.Series([os.path.basename(file) for file in file_names_test])
    df_test = df_import[df_import['name_img'].isin(file_test)]
    
    file_names_val = list(map(os.path.basename, list_val))
    # Filter the df based on the retained images
    file_val = pd.Series([os.path.basename(file) for file in file_names_val])
    df_val = df_import[df_import['name_img'].isin(file_val)]
    
    preparation(df_test,labels_test)
    preparation(df_val,labels_val)
    
    return


# Get df from image list
# Only returns rows which deals with the images in path_img
def get_df_notcatalog(path_img,df):
    next_df=pd.DataFrame()
    list_img=list(path_img.glob('**/*.jpg'))
    file_names = list(map(os.path.basename, list_img))
    file_names = pd.Series([os.path.basename(file) for file in list_img])
    next_df = df[df['name_img'].isin(file_names)]
    return(next_df)

# Changes the value of a label (the species)
def change_lbl(path_to_lbls):
    low_files=['train','test','val']
    
    for file in low_files:
        path_sub=os.path.join(path_to_lbls,file)        
        txt_files = os.listdir(path_sub)
        for txt in txt_files:
            mod_lines = []
            with open(os.path.join(path_sub,txt),'r') as e:
                lines = e.readlines()
                for line in lines:
                    line_2='0' + line[1:]
                    mod_lines.append(line_2)
                e.close()
                pass  # Empty file, nothing to write
            with open(os.path.join(path_sub,txt),'w') as g:
                g.writelines(mod_lines)

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
def superp_iou(bb1,bb2,s_err=None):
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
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        if iou == 0:
            return(None)
        if iou >= s_err:  # Adjust the IoU threshold as needed
            return True

# Tests if two bounding boxes are overlapping, then calculates iou
# Basically same as above
def superp_iou_min(bb1,bb2,s_err=None,nms=False):
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
    print('Unifying bounding boxes...')
    # For every image
    for i, image in enumerate(images):
        
        # Subset of orginal df with only image i
        tt_i = df[df['name_img'] == image]
        labels_i = tt_i['name_sp'].unique()
        progress = (i+1) * 100 // total
        if progress in [25, 50, 75, 100]:
            print_progress(progress)
        
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
                    if superp_iou_min(bb1,bb2,iou_thresh,nms)==True:
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
        progress = (group+1) * 100 // len(groupes)
        if progress in [25, 50, 75, 100]:
            print_progress(progress)
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
