#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

from PIL import Image, ImageDraw, ImageFilter, TiffTags
from skimage import measure, img_as_ubyte, color
from skimage.measure import ransac
from skimage.transform import rescale, resize, downscale_local_mean, warp, AffineTransform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.metrics.pairwise import euclidean_distances
import cv2
import json
import scanpy as sc
import anndata as ad
import shutil
import os
from tqdm import tqdm
import skimage
import matplotlib as mpl
import pickle
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import math
import numpy as np

Image.MAX_IMAGE_PIXELS = None
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams["figure.figsize"] = [6.4, 4.8]
mpl.rcParams.update({'font.size': 12})

from datetime import date
today = str(date.today())

import glob
from os import listdir
from os.path import isfile, join
from urllib.parse import urlparse

today = str(date.today())


## auxiallry functions 

def allignment_to_reference(
    props,
    Ref_path,
    scalefactors,
    path_output,
    sensitivity=2,
    min_s=4,
):
    """
        Parameters
        ----------     
        props 
            output from cellpose fiducial detection
        
        Ref_path
            path to auxilarry files used for registration
        scalefactors 
            general scale of images used for allignment 
        sensitivity
            threshold for similarity of matched spots. increasing number will inclrease matching accuracy but will reduce number of points for
            matching 
        min_s
            minimum samples requierd for ransac. use 3-4 for challanging images and 5 for better
    """

    # filter spots
    mpl.rcParams['figure.dpi'] = 150

    df = pd.DataFrame(props)
    print(df.head())
    # The formula for circularity is 4pi(area/perimeter^2)
    #To delete small regions...
    p = math.pi
    
    # arrange big spot Data
    C1 = np.array(df['centroid-0'].values) 
    C2 = np.array(df['centroid-1'].values)
    C = np.vstack((C1,C2)).T
    
    Corig = C.dot(scalefactors['scalefactor5K']).astype(int)
    Cdf = pd.DataFrame({'X': Corig[:, 0], 'Y': Corig[:, 1]})

    # distance between rows of X
    ED = euclidean_distances(C, C)
    EDT = ED[~np.eye(ED.shape[0],dtype=bool)].reshape(ED.shape[0],-1)
    MaxDist = EDT.max().max()
    EDT = np.sort(EDT.dot(1/MaxDist))
    EDT_Top10 = EDT[:,range(10)] # normalized distance to top 10 closest points for pattern recognition 
    print('found - '+str(ED.shape[0])+' Candidate points')
    # plt.plot(C1,C2,'b.', markersize=0.25)
    # plt.axis('off')
    # plt.show()
    PixelsPerMicrons = (scalefactors['scalefactor5K']*MaxDist)/10504
      #  Load pattern from reference data 
    LargeSpots = pd.read_csv(Ref_path+'Norm_Pattern_Dist_Flu_10x.csv')
    LargeSpotDistM = np.asmatrix(LargeSpots.to_numpy())
    Specificity = [None]*LargeSpotDistM.shape[0]
    PiaredIndex = []

    for point in range(LargeSpotDistM.shape[0]):
        SimilarityTo = 1/(euclidean_distances(np.asarray(LargeSpotDistM[point,3:53]),EDT_Top10))
        Specificity[point] = np.sort(SimilarityTo[0])[-1]/np.sort(SimilarityTo[0])[-2]
        if (Specificity[point]>sensitivity): # changed from 1.5 find pairs of patterns that match and are unique bteween image and reference 
            PiaredIndex.append([point,np.argmax(SimilarityTo[0])])
    print(PiaredIndex)    

    # Extrapolate to original scale and transform image to reference coordinates
    PiredIndexM = np.asmatrix(PiaredIndex)
    TargetLandmarks = np.array([C[PiredIndexM[:,1]]])[0][:,0]
    TargetLandmarksOriginal = TargetLandmarks.dot(scalefactors['scalefactor5K'])
    SourceLandmarksOriginal = LargeSpots[['X','Y']].to_numpy()
    SourceLandmarksOriginal = SourceLandmarksOriginal[PiredIndexM[:,0]][:,0]
    # find transformation matrix 
    print('Finding Transformation Matrix')
    print(PiaredIndex)
    model_robust, inliers = ransac((SourceLandmarksOriginal,TargetLandmarksOriginal), AffineTransform, min_samples=min_s,residual_threshold=10, max_trials=2000)
    M = np.asmatrix(np.array(model_robust.params))
    print(M)
    # import large and small point reference data 
    LargeSpots = pd.read_csv(Ref_path + 'fiducials_visium_v1_A.csv')
    SmallSpots = pd.read_csv(Ref_path + 'oligos_visium_v1_A_10x_new.csv')

    # transform grid to image space and plot ontop of original image
    LSM = np.asmatrix(np.vstack((np.array(LargeSpots['X'].values),np.array(LargeSpots['Y'].values),np.ones(LargeSpots['X'].values.shape[0]))).T)
    LSMT = np.matrix.transpose(LSM) 
    LSMTTarget = np.matmul(M,LSMT)
    SSM = np.asmatrix(np.vstack((np.array(SmallSpots['X'].values),np.array(SmallSpots['Y'].values),np.ones(SmallSpots['X'].values.shape[0]))).T)
    SSMT = np.matrix.transpose(SSM) 
    SSMTTarget = np.matmul(M,SSMT)

    # plot spot detection
    plt.clf() 
    plt.figure()
    plt.plot(SSMTTarget[1,:],SSMTTarget[0,:],'bo', markersize=1.5, fillstyle='none', markeredgewidth=0.1)
    plt.plot(LSMTTarget[1,:],LSMTTarget[0,:],'ro', markersize=3, fillstyle='none', markeredgewidth=0.1)
    plt.plot(C2*scalefactors['scalefactor5K'],C1*scalefactors['scalefactor5K'],'k.', markersize=0.25)
    mpl.rcParams['figure.dpi'] = 200
    plt.show()
   
    # correct a slight pixel shift that is caused due to differences in image (1 is 1st) vs array (0 is 1st) coordinates 
    LSMTTarget[0,:] = LSMTTarget[0,:]-(scalefactors['scalefactor5K'])
    SSMTTarget[0,:] = SSMTTarget[0,:]-(scalefactors['scalefactor5K'])
    LSMTTarget[1,:] = LSMTTarget[1,:]-(scalefactors['scalefactor5K'])
    SSMTTarget[1,:] = SSMTTarget[1,:]-(scalefactors['scalefactor5K'])
    
    
    mpl.rcParams['figure.dpi'] = 200
    imgP = Image.open(path_output+'/spatial/tissue_hires5K_image.png')
    plt.clf() 
    plt.figure()
    plt.imshow(imgP)
    plt.plot(SSMTTarget[1,:]*1/scalefactors['scalefactor5K'],
             SSMTTarget[0,:]*1/scalefactors['scalefactor5K'],
             'bo', markersize=1.5, fillstyle='none', markeredgewidth=0.1)
    plt.plot(LSMTTarget[1,:]*1/scalefactors['scalefactor5K'],
             LSMTTarget[0,:]*1/scalefactors['scalefactor5K'],
             'ro', markersize=3, fillstyle='none', markeredgewidth=0.1)
    plt.savefig(path_output+'spatial/aligned_fiducials.jpg')

   
    np.savetxt(path_output+'SSMTTarget.csv', SSMTTarget, delimiter=",")
    np.savetxt(path_output+'LSMTTarget.csv', LSMTTarget, delimiter=",")
    print('DONE!')
    return PixelsPerMicrons

        
def downsample_image(img, width, height, target_max_size=5000, greyscale=False):
    """
    Downsamples an image to a specified width and height, with an optional maximum target size.

    Parameters:
        img (numpy.ndarray): The input image to be downsampled.
        width (int): The desired width of the downsampled image.
        height (int): The desired height of the downsampled image.
        target_max_size (int, optional): The maximum size (width or height) of the downsampled image. Defaults to 5000.
        greyscale (bool, optional): Flag indicating whether to convert the downsampled image to greyscale. Defaults to False.

    Returns:
        tuple: A tuple containing the downsampled image (numpy.ndarray) and the scale factor used for downsampling (float).
    """
    scalefactor = max([width, height]) / target_max_size
    dim = (int(width * (1/scalefactor)), int(height * (1/scalefactor)))
    img_rescaled = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    if greyscale:
        img_rescaled = color.rgb2gray(img_rescaled)
    
    return img_rescaled, scalefactor

def align_and_extract(
    path_10x, 
    path_image, 
    ref_path,
    sensitivity=2, 
    use_10X_allignment = True,
    target_max_dimention = 5000,
    use_gpu = False,
):
    """
        Detect fiducials with cellpose model and affine transform reference grid to target image
        with torch GPU should run in less than a min on a standard laptop, CPU might take 5-6 min. 
        note that this alligner is more accurate but less robust than 10X's current alligner. for problematic tissues (e.g. fiducial edges covered by tissue) 
        please check buth use_10X_allignment = True or False and estimate allignment quality 
        Parameters
        ----------
        path_10x
            path to reference 10x data (fuicidals, small spots, barcodes)
        path_image
            path to image 
        ref_path 
            path to allignment fererence files 
        sensitivity
            Similarity threshold to reference fiducials, lower values will match more spots for the ransac model but might detect wrong spots, so play with it. between 0.7 - 4  
        use_10X_allignment = True
            if this option is selected registration is skipped and only the additional 5k resolution image is generated 
        
    """
    from cellpose import utils, io, models, plot

    # set up output directory if already exists remove and create
    os.chdir(path_10x)
    if not os.path.isdir('imagespot_preprocessing'): 
        os.makedirs('imagespot_preprocessing')
        os.makedirs('imagespot_preprocessing/spatial')
    else: 
        shutil.rmtree('imagespot_preprocessing')
        os.makedirs('imagespot_preprocessing')
        os.makedirs('imagespot_preprocessing/spatial')
        
    outputFolder = 'imagespot_preprocessing/' 
    
    Image.MAX_IMAGE_PIXELS = 10000000000 # bypass limitations 
    
    # copy raw h5 to output folder
    shutil.copyfile(path_10x+'/raw_feature_bc_matrix.h5',path_10x+'/imagespot_preprocessing/raw_feature_bc_matrix.h5')
    
    if 'ndpi' in path_image:
        slide = openslide.OpenSlide(path_image)
        slide.get_thumbnail(size = (600, 600))
        downsample_factor = min(slide.dimensions[0], slide.dimensions[1])/target_max_dimention
        downsample_level = slide.get_best_level_for_downsample(downsample_factor)
        levelx_img = slide.read_region((0,0), downsample_level, slide.level_dimensions[downsample_level])
        imgP = levelx_img.convert('RGB')
    else:
        imgP = Image.open(path_image) 

    width, height = imgP.size
    #width, height = imgP.shape[0], imgP.shape[1]
    img = np.array(imgP)
    print('Scaling images and saving') 
    # downsample Images and save output
    img_rescaled5K, scalefactor5K =  downsample_image(img,width,height,target_max_size=target_max_dimention,greyscale=False) 
    img_rescaled2K, scalefactor2K =  downsample_image(img,width,height,target_max_size=2000,greyscale=False)    
    img_rescaled600, scalefactor600 =  downsample_image(img,width,height,target_max_size=600,greyscale=False)    
    
    scalefactors = {'scalefactor5K':scalefactor5K,'scalefactor2K':scalefactor2K,'scalefactor600':scalefactor600,}

    plt.imsave(path_10x+ 'imagespot_preprocessing/spatial/tissue_hires5K_image.png',img_rescaled5K)
    plt.imsave(path_10x+ 'imagespot_preprocessing/spatial/tissue_hires_image.png',img_rescaled2K)
    plt.imsave(path_10x+ 'imagespot_preprocessing/spatial/tissue_lowres_image.png',img_rescaled600)

    if use_10X_allignment:
        print('Skipping custom allignment - using 10X coordinates')
        f = open(path_10x+'/spatial/scalefactors_json.json')
        json_dict = json.load(f)
        
        data = {
        'tissue_hires_scalef': 1/scalefactors['scalefactor2K'],
        'tissue_hires5K_scalef': 1/scalefactors['scalefactor5K'],
        'tissue_lowres_scalef': 1/scalefactors['scalefactor600'],
        'fiducial_diameter_fullres': json_dict['fiducial_diameter_fullres'],
        'spot_diameter_fullres': json_dict['spot_diameter_fullres'],
        }

        with open(path_10x+'imagespot_preprocessing/spatial/scalefactors_json.json', 'w') as outfile:
            json.dump(data, outfile) 
        
    else:
        # DEFINE CELLPOSE MODEL
        model = models.Cellpose(gpu=use_gpu, model_type='cyto')
        channels = [0,0]
        # you can run all in a list e.g.
        print('Running Cellpose model for fiducial detection')
        img_grey_rescaled, temp =  downsample_image(img,width,height,target_max_size=target_max_dimention,greyscale=True)    
        masks, flows, styles, diams = model.eval(img_grey_rescaled, diameter=55, channels=channels)
        props = measure.regionprops_table(masks, img_grey_rescaled, 
                                      properties=['label','area', 'equivalent_diameter', 'perimeter', 'centroid'])
        PixelsPerMicrons = allignment_to_reference(
            props,sensitivity=2,min_s=4,Ref_path=ref_path,
            scalefactors=scalefactors,path_output=path_10x+'imagespot_preprocessing/')
        generate_scale_factor(path=path_10x+'imagespot_preprocessing/spatial/',ppm=PixelsPerMicrons,scale_factors=scalefactors)

    print('Done!') 


def generate_scale_factor(
    path,
    ppm,
    scale_factors
):
    """
        Implemented to match 10X SpaceRanger josn file

        Parameters
        ----------
        path
            output directory.
        PixelsPerMicrons
            Pixesl per microns of the original resolution image.
        image_lowres_scalef
            scale factor for low resolution image.

    """     
    os.chdir(path)
    fiducial_diameter_fullres = ppm*85  
    spot_diameter_fullres = ppm*55
    data = {
        'tissue_hires_scalef': 1/scale_factors['scalefactor2K'],
        'tissue_hires5K_scalef': 1/scale_factors['scalefactor5K'],
        'tissue_lowres_scalef': 1/scale_factors['scalefactor600'],
        'fiducial_diameter_fullres': fiducial_diameter_fullres,
        'spot_diameter_fullres': spot_diameter_fullres,
    }
    with open('scalefactors_json.json', 'w') as outfile:
        json.dump(data, outfile)
        
# Detect tissue   
def detect_tissue_otsu(
    path,
    ref_path,
    spot_rad=1,
    sensitivity=1,
    Correcth5=True,
):
    """
        Detect tissue above a certain threshold, save and output in-tissue spot coordinates
        as well as an image of the tissue detected spots

        Parameters
        ----------
        path
            path to registration output directory
        Refpath
            path to reference 10x data (fuicidals, small spots, barcodes)
        sensitivity
            quantile threshold for tissue detection.
        spot_rad
            Spot radius to measure background, 1=55um, increase to indclude briader tissue context
        Correcth5
            when True a corrected AnnData object will be generated in the output folder (in addition to the "spatial" folder) with 3 levels of image resolutions:  
            1) hires5K - in additon to the SpaceRanger default this holds a higer resolution version defined by - max(imgae.shape) = 5000
            2) hires - identical to SpaceRanger hires image (2K resolution) 
            3) lowres - identical to SpaceRanger lowres image (0.6K resolution) 
            When false only the "spatial" folder would be produced in the end of the run with all its content 

    """
    SmallSpots = pd.read_csv(ref_path+'oligos_visium_v1_A_10x_new.csv')
    j = open(path+'imagespot_preprocessing/spatial/scalefactors_json.json'); data = json.load(j)
    tissue_lowres_scalef = data['tissue_lowres_scalef'] # Opening imaging metadata file
    spot_diameter_fullres = data['spot_diameter_fullres'] # Opening imaging metadata file
    img = np.array(Image.open(path+'imagespot_preprocessing/spatial/tissue_lowres_image.png'))
    SSMTTarget = genfromtxt(path+'imagespot_preprocessing/SSMTTarget.csv', delimiter=',')      
#     x = 0
    SSMTTarget = SSMTTarget*tissue_lowres_scalef
    sizeminiRad = round(spot_rad*spot_diameter_fullres*tissue_lowres_scalef)
    
#     crop_img = img[(SSMTTarget[0,x]-sizeminiRad*2).astype(int):(SSMTTarget[0,x]+sizeminiRad*2).astype(int),(SSMTTarget[1,x]-sizeminiRad*2).astype(int):(SSMTTarget[1,x]+sizeminiRad*2).astype(int)]
    
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu, thresh1 = cv2.threshold(imgG, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    
    print('Otsu threhold found to be - ' + str(otsu))
    d = {'SpotNum': [],'Barcode': [],'UnderTissue': [],'Col': [],'Row': [], 'X': [],'Y': [] }
    BarcodeTo_XY = pd.DataFrame(data=d)
    Points = range(0, SSMTTarget.shape[1], 1)
    
    i = 0
    for x in Points:  
        crop_img = img[(SSMTTarget[0,x]-sizeminiRad).astype(int):(SSMTTarget[0,x]+sizeminiRad).astype(int),(SSMTTarget[1,x]-sizeminiRad).astype(int):(SSMTTarget[1,x]+sizeminiRad).astype(int)]
        if (np.average(crop_img)<otsu*sensitivity):
            i = i+1
            df1 = pd.DataFrame(data={'SpotNum': [str(i).zfill(4)],'Barcode': SmallSpots['Barcode'][x],'UnderTissue': [1],'Col': SmallSpots['Col'][x],'Row': SmallSpots['Row'][x], 'X': SSMTTarget[0,x]/tissue_lowres_scalef,'Y': SSMTTarget[1,x]/tissue_lowres_scalef },index=[i-1])
            BarcodeTo_XY = pd.concat([BarcodeTo_XY,df1])
        else: 
            i = i+1
            df1 = pd.DataFrame(data={'SpotNum': [str(i).zfill(4)],'Barcode': SmallSpots['Barcode'][x],'UnderTissue': [0],'Col': SmallSpots['Col'][x],'Row': SmallSpots['Row'][x], 'X': SSMTTarget[0,x]/tissue_lowres_scalef,'Y': SSMTTarget[1,x]/tissue_lowres_scalef },index=[i-1])
            BarcodeTo_XY = pd.concat([BarcodeTo_XY,df1])
    
    BarcodeTo_XY = BarcodeTo_XY.drop(columns=['SpotNum'])
    BarcodeTo_XY.to_csv(path+'imagespot_preprocessing/spatial/tissue_positions_list.csv',header=False,index=False)
    # show tissue detection
    print(BarcodeTo_XY.head())
    plt.clf() 
    plt.figure()
    mpl.rcParams['figure.dpi'] = 200
    plt.imshow(img)
    plt.plot(BarcodeTo_XY.Y[BarcodeTo_XY.UnderTissue==1]*tissue_lowres_scalef,BarcodeTo_XY.X[BarcodeTo_XY.UnderTissue==1]*tissue_lowres_scalef,'go', markersize=2, markeredgewidth=0.1,alpha=0.5)
    plt.plot(BarcodeTo_XY.Y[BarcodeTo_XY.UnderTissue==0]*tissue_lowres_scalef,BarcodeTo_XY.X[BarcodeTo_XY.UnderTissue==0]*tissue_lowres_scalef,'ro', markersize=1, markeredgewidth=0.1,alpha=0.2)

    plt.savefig(path+'imagespot_preprocessing/spatial/detected_tissue_image.jpg')
    plt.show()
    if Correcth5:
        AnnVis = sc.read_visium(path+'imagespot_preprocessing/',count_file='raw_feature_bc_matrix.h5')
        img5k = np.array(Image.open(path+'imagespot_preprocessing/spatial/tissue_hires5K_image.png'))
        sample = list(AnnVis.uns['spatial'].keys())
        AnnVis.uns['spatial'][sample[0]]['images']['hires5K'] = img5k
        AnnVis.write_h5ad(path+'imagespot_preprocessing/corrected_raw_adata.h5ad')

    
    return BarcodeTo_XY

def generate_hires_grid(im, spot_diameter, pixels_per_micron):
    """
    Generates a grid of positions for high-resolution spots on an image.

    Parameters:
        im (numpy.ndarray): The input image.
        spot_diameter (float): The diameter of each spot in microns.
        pixels_per_micron (float): The number of pixels per micron in the image.

    Returns:
        numpy.ndarray: An array of positions representing the grid of spots.
    """
    import skimage
    
    helper = spot_diameter * pixels_per_micron
    X1 = np.linspace(helper, im.shape[0] - helper, round(im.shape[0] / helper))
    Y1 = np.linspace(helper, im.shape[1] - 2 * helper, round(im.shape[1] / (2 * helper)))
    X2 = X1 + spot_diameter / 2 * pixels_per_micron
    Y2 = Y1 + helper
    
    Gx1, Gy1 = np.meshgrid(X1, Y1)
    Gx2, Gy2 = np.meshgrid(X2, Y2)
    
    positions1 = np.vstack([Gy1.ravel(), Gx1.ravel()])
    positions2 = np.vstack([Gy2.ravel(), Gx2.ravel()])
    positions = np.hstack([positions1, positions2])
    
    return positions


def grid_anno(
    im,
    annotation_image_list,
    annotation_image_names,
    annotation_label_list,
    spot_diameter,
    ppm_in,
    ppm_out,
    
):
    """
        transfer annotations to spot grid 
        
        Parameters
        ---------- 
        im 
            Original image (for resizing annotations)
        annotation_image_list
            list of images with annotations in the form of integer corresponding to labels   
        annotation_image_names
            list of image names with annotations in the form of strings corresponding to images 
        annotation_label_list
            list of dictionaries to convert label data to morphology
        spot_diameter
            same diameter used for grid
        ppm_in 
            pixels per micron for input image
        ppm_out 
            used to scale xy grid positions to original image

    """
    print('generating grid with spot size - '+str(spot_diameter)+', with resolution of - '+str(ppm_in)+' ppm')
    positions = generate_hires_grid(im,spot_diameter,ppm_in)
    positions = positions.astype('float32')
    dim = [im.shape[0],im.shape[1]]
    # transform tissue annotation images to original size

    radius = spot_diameter/4 # measure the annotation from the center of the spot 
    df = pd.DataFrame(
        np.vstack((np.array(range(len(positions.T[:,0]))),positions.T[:,0],
                   positions.T[:,1])).T,
        columns=['index','x','y'])
    for idx0,anno in enumerate(annotation_image_list):
        anno_orig = skimage.transform.resize(anno,dim,preserve_range=True).astype('uint8') 
        anno_dict = {}
        number_dict = {}
        name = f'{anno=}'.split('=')[0]
        print(annotation_image_names[idx0])
        for idx1,pointcell in tqdm(enumerate(positions.T)):
            disk = skimage.draw.disk([int(pointcell[1]),int(pointcell[0])],radius,shape=anno_orig.shape)
            anno_dict[idx1] = annotation_label_list[idx0][int(np.median(anno_orig[disk]))]
            number_dict[idx1] = int(np.median(anno_orig[disk]))
        df[annotation_image_names[idx0]] = anno_dict.values()
        df[annotation_image_names[idx0]+'_number'] = number_dict.values()
    # scale to original image coordinates
    df['x'] = df['x']*ppm_out/ppm_in
    df['y'] = df['y']*ppm_out/ppm_in
    df['index'] = df['index'].astype(int)
    df.set_index('index', drop=True, inplace=True)
    return df

def bin_axis(ct_order, cutoff_values, df, axis_anno_name):
    """
    Bins a column of a DataFrame based on cutoff values and assigns manual bin labels.

    Parameters:
        ct_order (list): The order of manual bin labels.
        cutoff_values (list): The cutoff values used for binning.
        df (pandas.DataFrame): The DataFrame containing the column to be binned.
        axis_anno_name (str): The name of the column to be binned.

    Returns:
        pandas.DataFrame: The modified DataFrame with manual bin labels assigned.
    """
    # Manual annotations
    df['manual_bin_' + axis_anno_name] = 'unassigned'
    df['manual_bin_' + axis_anno_name] = df['manual_bin_' + axis_anno_name].astype('object')
    df.loc[np.array(df[axis_anno_name] < cutoff_values[0]), 'manual_bin_' + axis_anno_name] = ct_order[0]
    print(ct_order[0] + '= (' + str(cutoff_values[0]) + '>' + axis_anno_name + ')')
    
    for idx, r in enumerate(cutoff_values[:-1]):
        df.loc[np.array(df[axis_anno_name] >= cutoff_values[idx]) & np.array(df[axis_anno_name] < cutoff_values[idx+1]),
               'manual_bin_' + axis_anno_name] = ct_order[idx+1]
        print(ct_order[idx+1] + '= (' + str(cutoff_values[idx]) + '<=' + axis_anno_name + ') & (' + str(cutoff_values[idx+1]) + '>' + axis_anno_name + ')' )

    df.loc[np.array(df[axis_anno_name] >= cutoff_values[-1]), 'manual_bin_' + axis_anno_name] = ct_order[-1]
    print(ct_order[-1] + '= (' + str(cutoff_values[-1]) + '=<' + axis_anno_name + ')')

    df['manual_bin_' + axis_anno_name] = df['manual_bin_' + axis_anno_name].astype('category')
    df['manual_bin_' + axis_anno_name + '_int'] = df['manual_bin_' + axis_anno_name].cat.codes

    return df

def axis_3p_norm(
    Dist2ClusterAll,
    structure_list,
    df,
    axis_anno='cma_v3',
):
    df[axis_anno] = np.zeros(df.shape[0])
    counter = -1
    for b,c,a in zip(Dist2ClusterAll[structure_list[0]], Dist2ClusterAll[structure_list[1]],Dist2ClusterAll[structure_list[2]]):
        counter = counter+1
        if (b>=c): # meaning you are in the medulla 
            df[axis_anno].iloc[counter] = (int(b)-int(c))/(int(c)+int(b)) # non linear normalized distance between cortex with medulla for edge effect modulation 
        if (b<c): # meaning you are in the cortex 
            df[axis_anno].iloc[counter] = (int(a)-int(c)+0.5*int(b))/(int(a)+int(c)+0.5*int(b))-1 # shifted non linear distance between edge and medualla 
    return df 

    
def anno_to_visium_spots(
    df_spots,
    df_grid,
):
    
    print('make sure the coordinate systems are alligned e.g. axes are not flipped') 
    a = np.vstack([df_grid['x'],df_grid['y']])
    b = np.vstack([df_spots[5],df_spots[4]])
    plt.figure(dpi=100, figsize=[10,10])
    plt.title('spot space')
    plt.plot(b[0],b[1],'.', markersize=1)
    plt.show()
    plt.figure(dpi=100, figsize=[10,10])
    plt.plot(a[0],a[1],'.', markersize=1)
    plt.title('morpho spcae')
    plt.show()
    annotations = df_grid.columns[~df_grid.columns.isin(['x','y'])]
    import scipy
    # migrate continues annotations
    for k in annotations:
        print('migrating - '+k+' to segmentations')
        df_spots[k] = scipy.interpolate.griddata(points=a.T, values = df_grid[k], xi=b.T,method='nearest')
  
        
    return df_spots

def read_dapi(path):
    """
    Reads a DAPI image file.

    Parameters:
        path (str): The file path of the DAPI image.

    Returns:
        tuple: A tuple containing the image array (numpy.ndarray) and the pixels per micron (ppm) value (float).
    """
    im = Image.open(path)
    ppm = im.info['resolution'][0]
    return np.array(im), ppm

def cellpose_segmentation(im, ppm, model_type='cyto', gpu=True, diam_microns=8, min_area_microns=50):
    """
    Performs cell segmentation using the Cellpose library.

    Parameters:
        im (numpy.ndarray): The input image array.
        ppm (float): The pixels per micron value.
        model_type (str, optional): The type of model to use for segmentation. Defaults to 'cyto'.
        gpu (bool, optional): Flag indicating whether to use GPU acceleration. Defaults to True.
        diam_microns (float, optional): The diameter of cells in microns. Defaults to 8.
        min_area_microns (float, optional): The minimum area of cells in microns. Defaults to 50.

    Returns:
        numpy.ndarray: The segmented masks representing the cells.
    """
    from cellpose import models
    from skimage import morphology

    model = models.Cellpose(model_type=model_type, gpu=gpu)
    print('Starting cellpose segmentation')
    masks, flows, styles, diams = model.eval(
        x=im,
        diameter=int(ppm * diam_microns),
        resample=True,
        batch_size=1,
        flow_threshold=10,
        min_size=int(ppm * min_area_microns),
        cellprob_threshold=-1.5
    )
    print('Removing small masks')
    masks = morphology.remove_small_objects(masks, 2 * min_area_microns * ppm)
    labels = np.unique(masks)
    print('Found ' + str(len(labels.astype(int))) + ' segmentation labels')
    return masks



def channel_measurments(LabelImage, IntImage, marker):
    """
    Performs measurements on a channel image based on labeled regions.

    Parameters:
        LabelImage (numpy.ndarray): The labeled image.
        IntImage (numpy.ndarray): The intensity image.
        marker (str): The marker for the channel.

    Returns:
        pandas.DataFrame: A DataFrame containing the measured channel properties.
    """
    props = ('mean_intensity', 'max_intensity', 'area')
    ImageDatatmp = skimage.measure.regionprops_table(LabelImage, intensity_image=IntImage, properties=props, cache=True, separator='-')
    k_new = 'ch_' + marker + '_total_int'
    ImageDatatmp['area'] = ImageDatatmp['mean_intensity'] * ImageDatatmp['area']
    ImageDatatmp[k_new] = ImageDatatmp.pop('area')
    k_new = 'ch_' + marker + '_mean_int'
    ImageDatatmp[k_new] = ImageDatatmp.pop('mean_intensity')
    k_new = 'ch_' + marker + '_max_int'
    ImageDatatmp[k_new] = ImageDatatmp.pop('max_intensity')
    dfsegTemp = pd.DataFrame(data=ImageDatatmp)
    return dfsegTemp

def radius_mean(radius, ppm, data, image):
    """
    Calculates the mean intensity within a specified radius around each centroid.

    Parameters:
        radius (float): The radius in microns.
        ppm (float): The pixels per micron value.
        data (pandas.DataFrame): The DataFrame containing centroid coordinates.
        image (numpy.ndarray): The image.

    Returns:
        dict: A dictionary mapping indices to the mean intensities.
    """
    print('radius: ' + str(int(radius / ppm)))
    tempdict = {}
    for idx, c in enumerate(np.column_stack((data['centroid-1'].values, data['centroid-0'].values))):
        image[int(c[1]), int(c[0])]
        disk = skimage.draw.disk([int(c[1]), int(c[0])], radius, shape=image.shape)
        tempdict[data.index[idx]] = np.mean(image[disk])
    return tempdict

def add_channel_measurments(
    series_path,
    dfseg_nuc,
    meta_data,
    ppm,
    im_cell,
    add_perim=False,
    im_perim=None,
    add_radius=False,
    perif_r_microns=None
):
    """
    Adds channel measurements to the DataFrame of nuclear segmentations.

    Parameters:
        series_path (str): The path to the series.
        dfseg_nuc (pandas.DataFrame): The DataFrame of nuclear segmentations.
        meta_data (pandas.DataFrame): The metadata containing information about the channels.
        ppm (float): The pixels per micron value.
        im_cell (numpy.ndarray): The segmented nuclear image.
        add_perim (bool): Flag indicating whether to add measurements for the perimeter channel (default: False).
        im_perim (numpy.ndarray): The segmented perimeter image (default: None).
        add_radius (bool): Flag indicating whether to add measurements for radius segmentations (default: False).
        perif_r_microns (list): List of radius values in microns for radius segmentations (default: None).

    Returns:
        pandas.DataFrame: The updated DataFrame of nuclear segmentations with added channel measurements.
    """
    for idx, marker in enumerate(meta_data['protein']):
        # add channel data for the entire cell and add channel data for the perimeter
        if not meta_data['exclude'][idx]:
            print(marker)
            im = Image.open(series_path + meta_data['file'][idx])
            print(series_path + meta_data['file'][idx])
            print('mean before subtract - ' + str(np.mean(im)))
            im = np.array(im).astype('float64') - meta_data['threshold'][idx]
            im[np.where(im < 0)] = 0
            im = im.astype('uint8')
            print('mean after subtract - ' + str(np.mean(im)))
            dfsegTemp = channel_measurments(im_cell, im, marker)
            dfseg_nuc = pd.concat([dfseg_nuc, dfsegTemp], axis=1)
            # add perimeter segmentation
            if add_perim:
                dfsegTemp_prim = channel_measurments(im_perim, im, 'perim_' + marker)
                dfseg_nuc = pd.concat([dfseg_nuc, dfsegTemp_prim], axis=1)
            # add radius segmentations
            if add_radius:
                for r in perif_r_microns:
                    tempdict = radius_mean(radius=int(r * ppm), ppm=ppm, data=dfseg_nuc, image=im)
                    dfseg_nuc['ch_' + marker + '_r' + str(int(r))] = dfseg_nuc.index.map(tempdict)
        else:
            print('skipping - ' + marker)

    return dfseg_nuc


def generate_mask_grid(im, spot_diameter, pixels_per_micron):
    """
    Generate coordinates of a high resolution hexagonal grid and a label image. 
    
    Parameters
    ----------
    im : numpy.ndarray
        The input image to generate the grid for.
    spot_diameter : float
        Diameter of each spot in the hexagonal grid, in microns.
    pixels_per_micron : float
        Conversion factor from microns to pixels in the image.
        
    Returns
    -------
    positions : numpy.ndarray
        2D array where each column is the y, x coordinates of a spot center.
    label_image : numpy.ndarray
        2D label image of the same shape as `im`, where each spot is filled with a unique integer label.
    """
    import numpy as np
    from skimage.draw import disk

    helper = spot_diameter*pixels_per_micron
    X1 = np.linspace(helper, im.shape[0]-helper, int(np.round(im.shape[0]/helper)))
    Y1 = np.linspace(helper, im.shape[1]-2*helper, int(np.round(im.shape[1]/(2*helper))))
    X2 = X1 + spot_diameter/2*pixels_per_micron
    Y2 = Y1 + helper
    Gx1, Gy1 = np.meshgrid(X1,Y1)
    Gx2, Gy2 = np.meshgrid(X2,Y2)
    positions1 = np.vstack([Gy1.ravel(), Gx1.ravel()])
    positions2 = np.vstack([Gy2.ravel(), Gx2.ravel()])
    positions = np.hstack([positions1,positions2])
    
    # Create a label image
    label_image = np.zeros(im.shape, dtype=np.uint16)
    radius = (spot_diameter / 2) * pixels_per_micron
    for label, (x, y) in enumerate(positions.T, start=1):
        rr, cc = disk((y, x), radius, shape=im.shape)
        label_image[rr, cc] = label

    return positions, label_image    


def find_files(directory, query):
    """
    Search for files in a directory whose names contain a certain string and return them as a list.

    Parameters
    ----------
    directory : str
        The directory in which to search for files.
    query : str
        The string to search for within filenames.

    Returns
    -------
    list
        A list of full file paths that contain the query in their filename. 

    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if query in file:
                matching_files.append(os.path.join(root, file))
    return matching_files



def spot_pipeline(
    meta,
    path_to_images,
    marker_col,
    channel_col,
    include_col=None,
    nuclear_name='Hoechst',
    spot_diameter=50,
    plot_label=True,
):
    """
    Pipeline for spot detection and intensity measurements across multiple channels.

    Parameters
    ----------
    meta : pandas.DataFrame
        DataFrame containing metadata about the images. Expected to have columns
        specified by `marker_col` and `channel_col` parameters.
    path_to_images : str
        Path to the directory containing image files.
    marker_col : str
        Name of the column in `meta` that contains marker names.
    channel_col : str
        Name of the column in `meta` that contains channel numbers.
    include_col : str, optional
        Name of the column in 'meta' that determines if to extract the specificed channel
    nuclear_name : str, optional
        Name of the nuclear stain marker, by default 'Hoechst'.
    spot_diameter : float, optional
        Diameter of each spot in the hexagonal grid, in microns. By default 50.
    plot_label : bool, optional
        Whether to plot the label image, by default True.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with spot indices, centroid coordinates, and mean and max intensity
        for each channel. Each row corresponds to a spot, and columns correspond to
        properties of the spots.
    ppm : float32
        pixels per microns of the input images
    """

    from skimage.measure import regionprops_table

    # Find and read nuclear stain image
    first_nuclear = meta.loc[meta[marker_col] == nuclear_name, channel_col].iloc[0]
    nuclear_path = find_files(path_to_images, 'ch' + '{:02}'.format(int(first_nuclear)))
    im, ppm = read_dapi(nuclear_path[0])

    # Generate the grid and label image
    positions, label_image = generate_mask_grid(im, spot_diameter=spot_diameter, pixels_per_micron=ppm)
    

    # Measure the mean intensity in each labeled region in the nuclear image
    regions_nuclear = regionprops_table(label_image, intensity_image=im, properties=('label', 'mean_intensity'))
    df_nuclear = pd.DataFrame(regions_nuclear)
    df_nuclear.columns = ['spot_index', 'nuclear_mean_intensity']

    # Exclude spots with a mean intensity of 0 in the nuclear image
    df_nuclear = df_nuclear[df_nuclear['nuclear_mean_intensity'] != 0]

    # Initialize a DataFrame with spot indices and centroids
    regions = regionprops_table(label_image, properties=('label', 'centroid'))
    df = pd.DataFrame(regions)
    df.columns = ['spot_index', 'centroid_y', 'centroid_x']  # rename columns

    # Merge the nuclear and main dataframes, keeping only common rows (i.e., exclude empty spots)
    df = pd.merge(df, df_nuclear, on='spot_index')

    # For each channel, load the image and measure the mean and max intensity
    channels = np.array(meta[channel_col])
    for channel in channels:
        if include_col is None or meta[include_col][channel-1]:
            print('Reading image - ', meta.loc[meta[channel_col] == channel, marker_col].values)
            image_path = find_files(path_to_images, 'ch' + '{:02}'.format(int(channel)))
            im, _ = read_dapi(image_path[0])

            # Measure the mean and max intensity in each labeled region
            regions = regionprops_table(label_image, intensity_image=im, properties=('label', 'mean_intensity', 'max_intensity'))

            # Convert to a DataFrame and rename columns
            channel_df = pd.DataFrame(regions)
            name = np.array(meta.loc[meta[channel_col] == channel, marker_col])[0]

            # Check for duplicate column names and append a count if necessary
            name_base = name
            counter = 0
            while f'{name}_mean_intensity' in df.columns:
                name = f'{name_base}_{counter}'
                counter += 1

            channel_df.columns = ['spot_index', f'{name}_mean_intensity', f'{name}_max_intensity']

            # Merge with the main DataFrame
            df = pd.merge(df, channel_df, on='spot_index')
        else: 
            print('Excluding image - ', meta.loc[meta[channel_col] == channel, marker_col].values)

        
    if plot_label:
        plot_nuclear_channels(df, nuclear_name)
    
    return df, ppm


import numpy as np
import matplotlib.pyplot as plt

def plot_nuclear_channels(df, nuclear_name):
    """
    Plot all channels that include the provided nuclear_name.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing data to be plotted. Columns that include the nuclear_name
        will be selected for plotting.
    nuclear_name : str
        Name of the nuclear stain marker.
    """

    # Get all column names that include the nuclear_name as a substring
    nuclear_cols = [col for col in df.columns if nuclear_name in col]

    # Determine the layout of subplots (square layout, or as square as possible)
    num_plots = len(nuclear_cols)
    num_cols = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))

    # Create the subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    axs = axs.flatten()  # Flatten the axs array to simplify the indexing

    for i, col_name in enumerate(nuclear_cols):

        # Plot the data
        axs[i].plot(df[col_name])
        axs[i].set_title(col_name)

    # If there are less plots than subplots, delete the extra ones
    if num_rows * num_cols > num_plots:
        for i in range(num_plots, num_rows * num_cols):
            fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()

# Usage:
# plot_nuclear_channels(df, nuclear_name)





def plot_channel(df, channel_name):
    """
    Plots the specified channel data in x,y coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with spot indices, centroid coordinates, and intensity measurements.
    channel_name : str
        Name of the channel to plot.

    Returns
    -------
    None
    """
    # Check that the channel name is in the DataFrame
    if channel_name not in df.columns:
        print(f"Channel name '{channel_name}' not found in DataFrame.")
        return

    # Create scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(df['centroid_x'], df['centroid_y'], c=df[channel_name], cmap='viridis')
    plt.colorbar(label=channel_name)
    plt.gca().invert_yaxis()  # Optional: Invert y-axis if needed (common in image coordinates)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Plot of {channel_name}')
    plt.show()


def generate_3d_image(path_to_images, channel):
    """
    Function to generate a 3D numpy array from a sequence of 2D images.

    Parameters
    ----------
    path_to_images : str
        The path to the directory containing the images.
    channel : str
        The channel number to be read.

    Returns
    -------
    3d_image : numpy.ndarray
        The 3D image generated from the 2D images.
    """
    # Get list of image filenames
    filenames = [f for f in os.listdir(path_to_images) if f.endswith('.tif') and f'ch{channel:02}' in f]

    # Sort filenames by z-index
    filenames.sort(key=lambda f: int(f.split('_z')[1].split('.')[0]))

    # Initialize list to store 2D images
    images = []

    # Read each image as a numpy array and add to list
    for filename in filenames:
        img = Image.open(os.path.join(path_to_images, filename))
        img_array = np.array(img)
        images.append(img_array)

    # Stack 2D images into a 3D numpy array
    image_3d = np.stack(images, axis=0)

    return image_3d


# generte tiles from the original image of even size with overlap
def getTiles(img,tilesize=3500,Print=True):
    """
    tile the image for processing - from https://stackoverflow.com/questions/58383814/how-to-divide-an-image-into-evenly-sized-overlapping-if-needed-tiles 
    img - the image to tile 
    tilesize - max size of evenly created tiles
    """
    d,h,w = img.shape

    # Tile parameters
    wTile = tilesize
    hTile = tilesize

    # Number of tiles
    nTilesX = np.uint8(np.ceil(w / wTile))
    nTilesY = np.uint8(np.ceil(h / hTile))
    # Total remainders
    remainderX = nTilesX * wTile - w
    remainderY = nTilesY * hTile - h

    # Set up remainders per tile
    remaindersX = np.ones((nTilesX-1, 1)) * np.uint16(np.floor(remainderX / (nTilesX-1)))
    remaindersY = np.ones((nTilesY-1, 1)) * np.uint16(np.floor(remainderY / (nTilesY-1)))
    remaindersX[0:np.remainder(remainderX, np.uint16(nTilesX-1))] += 1
    remaindersY[0:np.remainder(remainderY, np.uint16(nTilesY-1))] += 1
    # Initialize array of tile boxes
    tiles = np.zeros((nTilesX * nTilesY, 4), np.uint16)
    # Determine proper tile boxes
    k = 0
    x = 0
    for i in range(nTilesX):
        y = 0
        for j in range(nTilesY):
            tiles[k, :] = (x, y, hTile, wTile)
            k += 1
            if (j < (nTilesY-1)):
                y = y + hTile - remaindersY[j]
        if (i < (nTilesX-1)):
            x = x + wTile - remaindersX[i]
    if Print:
        print('generated '+str(tiles.shape[0])+' tiles')
        # print('x y w h')
        print(tiles)
        
    return tiles

def tile_3d_segmentation(
    nuclear3D,
    tiles,
    ppm,
):
    # load models
    from cellpose import models
    model = models.Cellpose(model_type='cyto',gpu=True)
    labelImage = np.uint32(np.zeros(nuclear3D.shape))  # prepare label image as 32 bit 
    # segment over tiles with cellpose in 2D but with stitching 
    for idx,t in enumerate(tiles):
            Image = nuclear3D[:,t[1]:t[1]+t[3],t[0]:t[0]+t[2]] # crop image 
            print(Image.shape)
            if np.mean(Image[2,:,:])>1:            
                print('segmenting tile - '+str(idx)+', mean signal - '+str(np.mean(Image[2,:,:])))
                masks, flows, styles, diams = model.eval(Image,
                                                 diameter=int(3*ppm),
                                                 resample=True,
                                                 do_3D=False,
                                                 anisotropy=5,
                                                 cellprob_threshold=-10,
                                                 stitch_threshold=0.25, # changed from 0.5
                                                 flow_threshold=-6,
                                                 # progress=True,
                                                 # augment=True,
                                                 batch_size=1,
                                                )
                correctedlabels = np.nan_to_num(masks/masks)*(np.uint32(masks)+np.max(labelImage)) # corrected mask numbers according to last label in the image.
                correctededge = np.maximum(correctedlabels,labelImage[:,t[1]:t[1]+t[3],t[0]:t[0]+t[2]]) # take the new labels 
                labelImage[:,t[1]:t[1]+t[3],t[0]:t[0]+t[2]] = correctededge # add new lables to the full image 
                      
            else:
                print('skipping empty image')
                      
    return labelImage

def remove_touching_borders(image):
    """
    Removes voxels/pixels of a 2D or 3D label image (segmentation) that are "touching" borders of different labels.
    
    This function uses a neighborhood kernel to correlate the image, creating a 
    mask of voxels/pixels that have neighbors with a different label. It ignores 0 values 
    in the correlation and modifies a copy of the image, setting border voxels/pixels to 0.
    
    Parameters
    ----------
    image : ndarray
        The input image, typically a 2D or 3D array representing voxel/pixel intensities.
    
    Returns
    -------
    output_image : ndarray
        A copy of the input image with border voxels/pixels set to zero.
    

    Notes
    -----
    This function requires scipy.ndimage's correlate function.
    """
    
    from scipy.ndimage import correlate

    # Check the shape of the input array
    if len(image.shape) == 3:
        # Define the neighborhood kernel for 3D array
        kernel = np.ones((2, 2, 2))
    elif len(image.shape) == 2:
        # Define the neighborhood kernel for 2D array
        kernel = np.ones((2, 2))
    else:
        raise ValueError("Input array must be 2D or 3D")

    # Correlate the image with the kernel
    correlation = correlate(image, kernel)

    # Create a mask of pixels/voxels that have neighbors with a different label
    border_mask = correlation != kernel.sum() * image

    # Ignore 0 values in the correlation
    border_mask = border_mask & (image != 0)

    # Create a copy of the image to modify
    output_image = np.copy(image)

    # Set the border voxels/pixels to 0
    output_image[border_mask] = 0

    return output_image


def plot_random_color(img):
    """
    Assigns a random color to each unique label in the input image and plots the result.

    For 3D images, it plots each slice (along the first dimension) as a separate subplot. 
    For 2D images, it simply plots the colored image.
    
    Note: The first dimension in the input image is assumed to be the z-axis for 3D images.
    
    Parameters
    ----------
    img : ndarray
        The input image, which can be 2D (height, width) or 3D (depth, height, width) representing voxel/pixel labels.

    Notes
    -----
    This function requires matplotlib's pyplot (plt) module.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get the unique labels in the segmentation mask
    labels = np.unique(img)

    # Initialize an empty RGB image
    colored_img = np.zeros((*img.shape, 3), dtype=np.uint8)

    for label in labels:
        if label == 0:  # if considering 0 as background, skip it
            continue
        # generate a random color for each label
        color = np.random.randint(0, 255, size=3)
        # Apply the color to the regions of the label
        colored_img[img == label] = color

    if len(img.shape) == 3:
        # plot each slice as a separate subplot for 3D image
        fig = plt.figure()  # default figure size
        for i in range(colored_img.shape[0]):  # looping over the 1st dimension (z-axis)
            ax = fig.add_subplot(colored_img.shape[0]//2+1, 2, i+1)
            ax.imshow(colored_img[i, :, :])
            ax.axis('off')
            ax.set_aspect('equal')  # adjust the aspect to equal
        plt.tight_layout()  # to provide spacing between subplots
    elif len(img.shape) == 2:
        # plot the 2D image
        plt.figure()  # default figure size
        plt.imshow(colored_img)
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')  # adjust the aspect to equal

    plt.show()

    

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def cell_pipeline(
    masks,
    meta,
    path_to_images,
    marker_col,
    channel_col,
    ppm=1,
    include_col=None,
    nuclear_name='Hoechst',
    min_label_diameter=6,
    plot_label=True,
):
    """
    The function performs a pipeline for cell image processing and data extraction.

    Parameters
    ----------
    masks : ndarray
        A 3D numpy array of masks.
    meta : DataFrame
        Meta-data including channel and marker information.
    path_to_images : str
        Path to image data.
    marker_col : str
        Column name in meta DataFrame indicating the marker name.
    channel_col : str
        Column name in meta DataFrame indicating the channel number.
    ppm : float, optional
        Pixels per micron, for size conversion. Default is 1.
    include_col : str, optional
        Column name in meta DataFrame indicating whether to include the channel. Default is None.
    nuclear_name : str, optional
        Marker name for the nuclear channel. Default is 'Hoechst'.
    min_label_diameter : int, optional
        Minimum size of cells to include, in microns. Default is 6.
    plot_label : bool, optional
        Whether to plot label images. Default is True.

    Returns
    -------
    df : DataFrame
        A DataFrame containing the measured properties for each cell.
    """
    import skimage
    from skimage.measure import regionprops_table

    # plot label image 
    if plot_label:
        plot_mask(masks[1,:,:], title="Original mask", colorbar=True)
    
    # clean touching cells 
    masks = remove_touching_borders(masks)
    
    # plot cleaned mask
    if plot_label:
        plot_mask(masks[1,:,:], title="Mask after removing touching borders")
    
    # remove small masks
    masks = skimage.morphology.remove_small_objects(
        masks,
        min_size=ppm*min_label_diameter*min_label_diameter*min_label_diameter-1,
        connectivity=1)
    
    # plot cleaned mask
    if plot_label:
        plot_mask(masks[1,:,:], title="Mask after removing small objects")
    
    # measure general parameters
    props = ('label','centroid','area')
    GeneralDescriptors = skimage.measure.regionprops_table(masks, intensity_image=None, properties=props, cache=True)
    df = pd.DataFrame(data=GeneralDescriptors) # store in df 
    df = df.set_index('label')

    # Identify the nuclear channel
    nuclear_channel = meta.loc[meta[marker_col] == nuclear_name, channel_col].values[0]
    
    # Read the 3D image for nuclear channel
    nuclear_image = read_image_stack(path_to_images, f'*ch{nuclear_channel:02}_z*.tif')
    
    # Measure the mean and max intensity in each labeled region for nuclear channel
    nuclear_regions = regionprops_table(masks, intensity_image=nuclear_image, properties=('label', 'mean_intensity', 'max_intensity'))

    # Convert to a DataFrame and rename columns
    nuclear_df = pd.DataFrame(nuclear_regions)
    nuclear_df.columns = ['label', 'nuclear_mean_intensity', 'nuclear_max_intensity']

    # Merge with the main DataFrame
    df = pd.merge(df, nuclear_df, on='label')

    # Plot the mean intensity for nuclear channel
    plot_scatter(df['centroid-1'], df['centroid-2'], c=df['nuclear_mean_intensity'], title=f'Mean intensity of {nuclear_name} channel', colorbar_label='Mean intensity')

    # For each non-nuclear channel, load the 3D image and measure the mean and max intensity
    channels = np.array(meta[channel_col])
    for channel in channels:
        if channel == nuclear_channel:
            continue
        if include_col is None or meta[include_col][channel-1]:
            print('Reading images - ', meta.loc[meta[channel_col] == channel, marker_col].values)
            # Read the 3D image for this channel
            image_stack = read_image_stack(path_to_images, f'*ch{channel:02}_z*.tif')

            # Measure the mean and max intensity in each labeled region
            regions = regionprops_table(masks, intensity_image=image_stack, properties=('label', 'mean_intensity', 'max_intensity'))

            # Convert to a DataFrame and rename columns
            channel_df = pd.DataFrame(regions)
            name = np.array(meta.loc[meta[channel_col] == channel, marker_col])[0]

            # Check for duplicate column names and append a count if necessary
            name_base = name
            counter = 0
            while f'{name}_mean_intensity' in df.columns:
                name = f'{name_base}_{counter}'
                counter += 1

            channel_df.columns = ['label', f'{name}_mean_intensity', f'{name}_max_intensity']

            # Merge with the main DataFrame
            df = pd.merge(df, channel_df, on='label')
            
            # Plot the mean intensity for this channel
            plot_scatter(df['centroid-1'], df['centroid-2'], c=df[f'{name}_mean_intensity'], title=f'Mean intensity of {name} channel', colorbar_label='Mean intensity')

        else: 
            print('Excluding images - ', meta.loc[meta[channel_col] == channel, marker_col].values)

    return df


def read_image_stack(dir_path, filename_pattern):
    import fnmatch
    from skimage.io import imread

    # Get a sorted list of file paths matching the filename_pattern
    file_paths = sorted(os.path.join(dir_path, fn) for fn in os.listdir(dir_path) if fnmatch.fnmatch(fn, filename_pattern))
    
    # Load each 2D image into a list
    image_list = [imread(fp) for fp in file_paths]
    
    # Convert the list of 2D images into a 3D array
    image_stack = np.stack(image_list)
    
    return image_stack


def plot_mask(mask, title=None, cmap=None, figsize=(25,25), colorbar=False):
    plt.figure(figsize=figsize)
    plt.imshow(mask)
    if colorbar:
        plt.colorbar()
    if title:
        plt.title(title)
    plt.show()

    
def plot_scatter(x, y, c, title=None, cmap=cm.viridis, figsize=(10,10), colorbar_label=None):
    plt.figure(figsize=figsize)
    plt.scatter(x, y, c=c, cmap=cmap,s=0.5)
    if title:
        plt.title(title)
    if colorbar_label:
        plt.colorbar(label=colorbar_label)
    plt.show()

    
def calculate_axis_3p(df_ibex, anno, structure, output_col, w=[0.5,0.5], prefix='L2_dist_'):
    """
    Function to calculate a unimodal nomralized axis based on ordered structure of S1 -> S2 -> S3.

    Parameters:
    -----------
    df_ibex : DataFrame
        Input DataFrame that contains the data.
    anno : str, optional
        Annotation column. 
    structure : list of str, optional
        List of structures to be meausure. [S1, S2, S3]
    w : list of float, optional
        List of weights between the 2 components of the axis w[0] * S1->S2 and w[1] * S2->S3. Default is [0.2,0.8].
    prefix : str, optional
        Prefix for the column names in DataFrame. Default is 'L2_dist_'.
    output_col : str, optional
        Name of the output column.

    Returns:
    --------
    df : DataFrame
        DataFrame with calculated new column.
    """
    df = df_ibex.copy()
    a1 = (df[prefix + anno +'_'+ structure[0]] - df[prefix + anno +'_'+ structure[1]]) \
    /(df[prefix + anno +'_'+ structure[0]] + df[prefix + anno +'_'+ structure[1]])
    
    a2 = (df[prefix + anno +'_'+ structure[1]] - df[prefix + anno +'_'+ structure[2]]) \
    /(df[prefix + anno +'_'+ structure[1]] + df[prefix + anno +'_'+ structure[2]])
    df[output_col] = w[0]*a1 + w[1]*a2
    
    return df


def calculate_axis_2p(df_ibex, anno, structure, output_col, prefix='L2_dist_'):
    """
    Function to calculate a unimodal nomralized axis based on ordered structure of S1 -> S2 .

    Parameters:
    -----------
    df_ibex : DataFrame
        Input DataFrame that contains the data.
    anno : str, optional
        Annotation column. 
    structure : list of str, optional
        List of structures to be meausure. [S1, S2]
    prefix : str, optional
        Prefix for the column names in DataFrame. Default is 'L2_dist_'.
    output_col : str, optional
        Name of the output column.

    Returns:
    --------
    df : DataFrame
        DataFrame with calculated new column.
    """
    df = df_ibex.copy()
    a1 = (df[prefix + anno +'_'+ structure[0]] - df[prefix + anno +'_'+ structure[1]]) \
    /(df[prefix + anno +'_'+ structure[0]] + df[prefix + anno +'_'+ structure[1]])

    df[output_col] = a1 
    
    return df


def load_and_combine_annotations(folder, file_names, spot_diameter, load_colors=True):
    """
    Load tissue annotations from multiple files and combine them into a single DataFrame.

    Parameters
    ----------
    folder : str
        Folder path where the annotation files are stored.
    file_names : list of str
        List of names of the annotation files.
    spot_diameter : int
        Diameter of the spots.
    load_colors : bool, optional
        Whether to load colors. Default is True.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame that combines all the loaded annotations.
    ppm_grid : float
        Pixels per micron for the grid of the last loaded annotation.
    """
    df_list = []
    ppm_grid = None

    for file_name in file_names:
        df, ppm_grid = anno_to_grid(folder=folder, file_name=file_name, spot_diameter=spot_diameter, load_colors=load_colors)
        df_list.append(df)

    # Concatenate all dataframes
    df = pd.concat(df_list, join='inner', axis=1)

    # Remove duplicated columns
    df = df.loc[:, ~df.columns.duplicated()].copy()

    return df, ppm_grid


def anno_to_grid(folder, file_name, spot_diameter, load_colors=False,null_number=1):
    """
    Load annotations and transform them into a spot grid.
    
    Parameters
    ----------
    folder : str
        Folder path for annotations.
    file_name : str
        Name for tif image and pickle without extensions.
    spot_diameter : float
        The diameter used for grid.
    load_colors : bool, optional
        If True, get original colors used for annotations. Default is False.
    null_numer : int
        value of the label image where no useful information is stored e.g. background or unassigned pixels (usually 0 or 1). Default is 1

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with the grid annotations.
    """
    
    im, anno_order, ppm, anno_color = load_annotation(folder, file_name, load_colors)

    df = grid_anno(
        im,
        [im],
        [file_name],
        [anno_order],
        spot_diameter,
        ppm,
        ppm,
    )

    return df,ppm




def load_annotation(folder, file_name, load_colors=False):
    """
    Loads the annotated image from a .tif file and the translation from annotations 
    to labels from a pickle file.

    Parameters
    ----------
    folder : str
        Folder path for annotations.
    file_name : str
        Name for tif image and pickle without extensions.
    load_colors : bool, optional
        If True, get original colors used for annotations. Default is False.

    Returns
    -------
    tuple
        Returns annotation image, annotation order, pixels per microns, and annotation color.
        If `load_colors` is False, annotation color is not returned.
    """
    
    imP = Image.open(folder + file_name + '.tif')

    ppm = imP.info['resolution'][0]
    im = np.array(imP)

    print(f'loaded annotation image - {file_name} size - {str(im.shape)}')
    with open(folder + file_name + '.pickle', 'rb') as handle:
        anno_order = pickle.load(handle)
        print('loaded annotations')        
        print(anno_order)
    with open(folder + file_name + '_ppm.pickle', 'rb') as handle:
        ppm = pickle.load(handle)
        print('loaded ppm')        
        print(ppm)
        
    if load_colors:
        with open(folder + file_name + '_colors.pickle', 'rb') as handle:
            anno_color = pickle.load(handle)
            print('loaded color annotations')        
            print(anno_color)
        return im, anno_order, ppm['ppm'], anno_color
    
    else:
        return im, anno_order, ppm['ppm']

    
def dist2cluster_fast(df, annotation, KNN=5, logscale=False):
    from scipy.spatial import cKDTree

    print('calculating distance matrix with cKDTree')

    points = np.vstack([df['x'],df['y']]).T
    categories = np.unique(df[annotation])

    Dist2ClusterAll = {c: np.zeros(df.shape[0]) for c in categories}

    for idx, c in enumerate(categories):
        indextmp = df[annotation] == c
        if np.sum(indextmp) > KNN:
            print(c)
            cluster_points = points[indextmp]
            tree = cKDTree(cluster_points)
            # Get KNN nearest neighbors for each point
            distances, _ = tree.query(points, k=KNN)
            # Store the mean distance for each point to the current category
            if KNN == 1:
                Dist2ClusterAll[c] = distances # No need to take mean if only one neighbor
            else:
                Dist2ClusterAll[c] = np.mean(distances, axis=1)

    for c in categories:              
        if logscale:
            df["L2_dist_log10_"+annotation+'_'+c] = np.log10(Dist2ClusterAll[c])
        else:
            df["L2_dist_"+annotation+'_'+c] = Dist2ClusterAll[c]

    return Dist2ClusterAll



def map_annotations_to_target(df_source, df_target, ppm_source,ppm_target, plot=True, how='nearest', max_distance=50):
    """
    map annotations to any form of of csv where you have x y cooodinates spot df (cells or spots) data with high-resolution grid.
    note! - xy coordinates must be named 'x' and 'y'

    Parameters
    ----------
    df_source : pandas.DataFrame
        Dataframe with grid data.
    df_target : pandas.DataFrame
        Dataframe with target data.
    ppm_source : float 
        Pixels per micron of source data.
    ppm_target : float 
        Pixels per micron of target data.
    plot : bool, optional
        If true, plots the coordinates of the grid space and the spot space to make sure they are aligned. Default is True.
    how : string, optinal
        This determines how the association between the 2 grids is made from the scipy.interpolate.griddata function. Default is 'nearest'
        if the data is categorial then only the 'nearest' would work but if interpolation is needed one should supbset to only numeric data.
    max_distance : int
        Factor to calculate maximal distance where points are not migrated. The final max_distance used will be max_distance * ppm_target.
   
    Returns
    -------
    df_target : pandas.DataFrame
        Annotated dataframe with additional annotations from the source data.
    """
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    
    # generate matched coordinate space 
    a = np.vstack([df_source['x']/ppm_source, df_source['y']/ppm_source])
    b = np.vstack([df_target['x']/ppm_target, df_target['y']/ppm_target])
    
    if plot:
        print('Make sure the coordinate systems are aligned, e.g., axes are not flipped and the resolution is matched.') 
        plt.figure(dpi=100, figsize=[10, 10])
        plt.title('Target space')
        plt.plot(b[0], b[1], '.', markersize=1)
        plt.show()
        
        plt.figure(dpi=100, figsize=[10, 10])
        plt.plot(a[0], a[1], '.', markersize=1)
        plt.title('Source space')
        plt.show()

    annotations = df_source.columns[~df_source.columns.isin(['x', 'y'])] # extract annotation categories
    
    for k in annotations:
        print('Migrating source annotation - ' + k + ' to target space.')
        
        # Interpolation
        df_target[k] = griddata(points=a.T, values=df_source[k], xi=b.T, method=how)
        
        # Create KDTree
        tree = cKDTree(a.T)
        
        # Query tree for nearest distance
        distances, _ = tree.query(b.T, distance_upper_bound=max_distance)
        
        # Mask df_spots where the distance is too high
        df_target[k][distances==np.inf] = None
  
    return df_target





def read_visium_table(vis_path):
    """
    This function reads a scale factor from a json file and a table from a csv file, 
    then calculates the 'ppm' value and returns the table with the new column names.
    
    note that for CytAssist the header is changes so this funciton should be updated
    """
    with open(vis_path + '/spatial/scalefactors_json.json', 'r') as f:
        scalef = json.load(f)

    ppm = scalef['spot_diameter_fullres'] / 55 

    df_visium_spot = pd.read_csv(vis_path + '/spatial/tissue_positions_list.csv', header=None)

    df_visium_spot.rename(columns={4:'y',5:'x',1:'in_tissue',0:'barcode'}, inplace=True)
    df_visium_spot.set_index('barcode', inplace=True)

    return df_visium_spot, ppm



# get morphology 

import math
import time
import numpy as np
import pandas as pd
from scipy import fftpack
from skimage import measure
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import gabor
from skimage.util import img_as_ubyte
from skimage.util import img_as_ubyte
import multiprocessing as mp
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft2
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage.measure import regionprops



def calculate_features_for_label(label, regionmask, a, distances, angles, props):
    mask = (regionmask == label)  # Create mask for the current cell
    a_masked = a.copy()
    a_masked[~mask] = 0  # Mask the intensity image
    comatrix = graycomatrix(a_masked, distances=distances, angles=angles, levels=256)

    features = {'label': label}  # Initialize features dict for the current cell
    for p in props:
        tmp_features = graycoprops(comatrix, prop=p).flatten()  # Flattened to handle multiple distances/angles
        for idx, feature in enumerate(tmp_features):
            features[f"texture_{p}_{idx}"] = feature
    return features

def texture_features(regionmask, intensity_image):
    unique_labels = np.unique(regionmask)  # List of unique cell labels
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background (label 0)

    a = img_as_ubyte(intensity_image.copy())
    if a.shape != regionmask.shape:
        print(f'Shapes do not match: a.shape={a.shape}, regionmask.shape={regionmask.shape}')
        return []

    angles = (0, np.pi / 4, np.pi / 3)
    props = ("contrast", "dissimilarity", "correlation", "ASM")
#     props = ("contrast", "energy")
    distances = (1,)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        features_list = pool.starmap(calculate_features_for_label, [(label, regionmask, a, distances, angles, props) for label in unique_labels])

    return features_list


def shape_features(im_cell,im=None):
    labels = np.unique(im_cell)[1:]  # Exclude background label
    shape_features_list = []
    
    for label in labels:
        mask = (im_cell == label)
        props = regionprops(mask.astype(int))[0]  # Extract properties of the region

        shape_features = {'label': label,
                          'area': props.area,
                          'solidity': props.solidity,
                          'perimeter': props.perimeter,
                          'extent': props.extent,
                          'eccentricity': props.eccentricity,
                          'major_axis_length': props.major_axis_length,
                          'minor_axis_length': props.minor_axis_length}
        shape_features_list.append(shape_features)
    
    return shape_features_list



# Intensity Features
def intensity_features(regionmask, intensity_image):
    unique_labels = np.unique(regionmask)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    intensity_features_list = []
    for label in unique_labels:
        mask = (regionmask == label)
        masked_intensity = intensity_image[mask]
        intensity_features = {'label': label,
                              'mean_intensity': np.mean(masked_intensity),
                              'median_intensity': np.median(masked_intensity),
                              'std_intensity': np.std(masked_intensity),
                              'skewness_intensity': skew(masked_intensity),
                              'kurtosis_intensity': kurtosis(masked_intensity)}
        intensity_features_list.append(intensity_features)
    return intensity_features_list

# Frequency-domain Features
def frequency_features(regionmask, intensity_image):
    props = measure.regionprops(regionmask, intensity_image)
    frequency_features_list = []
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        masked_intensity = intensity_image[minr:maxr, minc:maxc]
        f_transform = fft2(masked_intensity)  # Fourier Transform
        f_spectrum = np.abs(f_transform)**2  # Power Spectrum
        frequency_features = {'label': prop.label,
                              'mean_spectrum': np.mean(f_spectrum),
                              'median_spectrum': np.median(f_spectrum),
                              'std_spectrum': np.std(f_spectrum)}
        frequency_features_list.append(frequency_features)
    return frequency_features_list


# LBP Features
def lbp_features(regionmask, intensity_image, P=8, R=1):
    props = measure.regionprops(regionmask, intensity_image)
    lbp_features_list = []
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        masked_intensity = intensity_image[minr:maxr, minc:maxc]
        lbp = local_binary_pattern(masked_intensity, P, R)
        lbp_features = {'label': prop.label,
                        'mean_lbp': np.mean(lbp),
                        'std_lbp': np.std(lbp)}
        lbp_features_list.append(lbp_features)
    return lbp_features_list


# Gabor Features
def gabor_features(regionmask, intensity_image, frequency=0.6):
    props = measure.regionprops(regionmask, intensity_image)
    gabor_features_list = []
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        masked_intensity = intensity_image[minr:maxr, minc:maxc]
        gabor_response = gabor(masked_intensity, frequency=frequency)
        gabor_features = {'label': prop.label,
                          'mean_gabor': np.mean(gabor_response),
                          'std_gabor': np.std(gabor_response)}
        gabor_features_list.append(gabor_features)
    return gabor_features_list



def get_all_properties_2d(im_cell, im, feature_list=None):
    """
    Function to extract all features from 2D image.
    Arguments:
    im_cell -- input image
    im -- intensities
    feature_list -- list of features to be calculated, default is all features
    Returns:
    dfseg_nuc -- a dataframe of features calculated for each label
    """

    if feature_list is None:
        feature_list = ["shape", "texture", "intensity", "frequency", "lbp", "gabor"]

    # Get the labels of the cells
    labels = np.unique(im_cell)
    dfseg_nuc = pd.DataFrame(labels, columns=['label'])

    feature_dict = {"shape": shape_features, "texture": texture_features, "intensity": intensity_features,
                    "frequency": frequency_features, "lbp": lbp_features, "gabor": gabor_features}

    for feature in feature_list:
        if feature in feature_dict:
            start = time.time()
            feature_df = pd.DataFrame(feature_dict[feature](im_cell, im)).set_index('label')
            end = time.time()
            print(f"{feature.capitalize()} features: {end - start} seconds, {len(feature_df.columns)} features")
            dfseg_nuc = pd.concat([dfseg_nuc, feature_df], axis=1)
        else:
            print(f"Invalid feature: {feature}")
    return dfseg_nuc.reset_index()


from collections import defaultdict
import pandas as pd
import numpy as np

# Your features are already defined above
# Now let's define feature_dict with your features and corresponding functions
feature_dict = {
    'texture': texture_features,
    'shape': shape_features,
    'intensity': intensity_features,
    'frequency': frequency_features,
    'lbp': lbp_features,
    'gabor': gabor_features,
    # add more features here
}

import time

def get_all_properties_3d(im_cell, im, feature_list=feature_dict.keys(), dim_order='xyz'):
    if dim_order.lower() not in ['xyz', 'zyx']:
        raise ValueError('Invalid dim_order, choose either "xyz" or "zyx"')

    z_slices = im_cell.shape[2] if dim_order.lower() == 'xyz' else im_cell.shape[0]
    feature_dataframes = []

    for z in range(z_slices):
        if dim_order.lower() == 'zyx':
            im_cell_slice, im_slice = im_cell[z, :, :], im[z, :, :]
        else:  # 'xyz'
            im_cell_slice, im_slice = im_cell[:, :, z], im[:, :, z]
            
        slice_labels = np.unique(im_cell_slice)
        # Exclude background (0)
        slice_labels = slice_labels[slice_labels != 0]

        # Calculate features for each label in the slice
        for feature in feature_list:
            if feature in feature_dict:
                start = time.time()
                feature_df = pd.DataFrame(feature_dict[feature](im_cell_slice, im_slice)).set_index('label')
                feature_df['z_slice'] = z  # add z slice information
                end = time.time()
                print(f"{feature.capitalize()} features for slice {z}: {end - start} seconds, {len(feature_df.columns)} features")

                # Append feature dataframe for each label
                for label in slice_labels:
                    if label in feature_df.index:
                        feature_dataframes.append(feature_df.loc[[label]])
            else:
                print(f"Invalid feature: {feature}")

    # Create a multi-level index dataframe
    multi_index_df = pd.concat(feature_dataframes)

    # Compute min, max, sum, and mean of features for each label (excluding 'z_slice')
    cols = multi_index_df.columns.difference(['z_slice'])
    df_min = multi_index_df[cols].groupby('label').min().add_suffix('_min')
    df_max = multi_index_df[cols].groupby('label').max().add_suffix('_max')
    df_sum = multi_index_df[cols].groupby('label').sum().add_suffix('_sum')
    df_mean = multi_index_df[cols].groupby('label').mean().add_suffix('_mean')

    # Concatenate the min, max, sum, and mean dataframes
    df_final = pd.concat([df_min, df_max, df_sum, df_mean], axis=1)

    return df_final.reset_index()


# def dot_plot_anno(adata, cell_type_anno, cell_calling_certainty_anno, fraction, cells, groupby, **kwargs):
#     """
#     Creates a dot plot with scanpy using data from an AnnData object, after applying a filter on cell calling certainty.

#     Parameters
#     ----------
#     adata : anndata.AnnData
#         The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
#         to cells and columns to genes.
#     cell_type_anno : str
#         The name of the column in `adata.obs` that contains the cell type annotations.
#     cell_calling_certainty_anno : str
#         The name of the column in `adata.obs` that contains the cell calling certainty values.
#     fraction : float
#         The minimum cell calling certainty to include a cell in the plot.
#     cells : list of str
#         The list of cell types to be plotted.
#     groupby : str
#         The key of the observation grouping to consider.
#     **kwargs : dict
#         Additional keyword arguments are passed to `scanpy.pl.dotplot`.

#     Returns
#     -------
#     None. Displays a dot plot visualization.
#     """
#     adata_filt = adata[adata.obs[cell_calling_certainty_anno] > fraction]
    
#     # Generate a new AnnData object
#     def generate_anndata(adata, cell_type_anno, cell_calling_certainty_anno):
#         # Get unique cell types
#         unique_cell_types = adata.obs[cell_type_anno].unique()
#         # Initialize a new data array with zeros
#         data = np.zeros((adata.shape[0], len(unique_cell_types)))
#         # Loop over unique cell types
#         for i, cell_type in enumerate(unique_cell_types):
#             # Get indices where cell type is the current cell type
#             indices = np.where(adata.obs[cell_type_anno] == cell_type)[0]
#             # Get corresponding certainties
#             certainties = adata.obs.iloc[indices][cell_calling_certainty_anno].values
#             # Assign these values to the corresponding column in data
#             data[indices, i] = certainties
            
#         # Create new AnnData object
#         adata_new = ad.AnnData(X=data,
#                                     obs=adata.obs,
#                                     var=pd.DataFrame(index=unique_cell_types))
#         return adata_new

#     plot_adata = generate_anndata(adata_filt, cell_type_anno, cell_calling_certainty_anno)
    
   
#     # Set figure parameters
# #     sc.set_figure_params(figsize=[4,4])
    
#     # Create dotplot
#     sc.pl.dotplot(plot_adata, var_names=cells, groupby=groupby, **kwargs)

    
def dot_plot_anno(adata, cell_type_anno, cell_calling_certainty_anno, fraction, cells, groupby, **kwargs):
    """
    Creates a dot plot with scanpy using data from an AnnData object, after applying a filter on cell calling certainty.

    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    cell_type_anno : str
        The name of the column in `adata.obs` that contains the cell type annotations.
    cell_calling_certainty_anno : str
        The name of the column in `adata.obs` that contains the cell calling certainty values.
    fraction : float
        The minimum cell calling certainty to include a cell in the plot.
    cells : list of str
        The list of cell types to be plotted.
    groupby : str
        The key of the observation grouping to consider.
    **kwargs : dict
        Additional keyword arguments are passed to `scanpy.pl.dotplot`.

    Returns
    -------
    None. Displays a dot plot visualization.
    """
    adata_filt = adata[adata.obs[cell_calling_certainty_anno] > fraction]
    
    # Generate a new AnnData object
    def generate_anndata(adata, cell_type_anno, cell_calling_certainty_anno, cells):
        # Get unique cell types
        unique_cell_types = list(set(cells).union(set(adata.obs[cell_type_anno].unique())))
        # Initialize a new data array with zeros
        data = np.zeros((adata.shape[0], len(unique_cell_types)))
        # Loop over unique cell types
        for i, cell_type in enumerate(unique_cell_types):
            if cell_type in adata.obs[cell_type_anno].unique():
                # Get indices where cell type is the current cell type
                indices = np.where(adata.obs[cell_type_anno] == cell_type)[0]
                # Get corresponding certainties
                certainties = adata.obs.iloc[indices][cell_calling_certainty_anno].values
                # Assign these values to the corresponding column in data
                data[indices, i] = certainties
            
        # Create new AnnData object
        adata_new = ad.AnnData(X=data,
                                    obs=adata.obs,
                                    var=pd.DataFrame(index=unique_cell_types))
        return adata_new

    plot_adata = generate_anndata(adata_filt, cell_type_anno, cell_calling_certainty_anno, cells)
    
    # Check if all cell types in 'cells' are present in 'plot_adata'
    missing_cells = list(set(cells) - set(plot_adata.var_names))
    
    # Raise a warning if some cell types are missing
    if missing_cells:
        print(f"Warning: the following cell types are not found in the data and will be plotted as empty columns: {missing_cells}")
    
    # Set figure parameters
    # sc.set_figure_params(figsize=[4,4])
    
    # Create dotplot
    sc.pl.dotplot(plot_adata, var_names=cells, groupby=groupby, **kwargs)

    
    
def matrix_plot_anno(adata, cell_type_anno, cell_calling_certainty_anno, fraction, cells, groupby, **kwargs):
    """
    Creates a matrix plot with scanpy using data from an AnnData object, after applying a filter on cell calling certainty.

    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    cell_type_anno : str
        The name of the column in `adata.obs` that contains the cell type annotations.
    cell_calling_certainty_anno : str
        The name of the column in `adata.obs` that contains the cell calling certainty values.
    fraction : float
        The minimum cell calling certainty to include a cell in the plot.
    cells : list of str
        The list of cell types to be plotted.
    groupby : str
        The key of the observation grouping to consider.
    **kwargs : dict
        Additional keyword arguments are passed to `scanpy.pl.matrixplot`.

    Returns
    -------
    None. Displays a matrix plot visualization.
    """
    adata_filt = adata[adata.obs[cell_calling_certainty_anno] > fraction]
    
    # Generate a new AnnData object
    def generate_anndata(adata, cell_type_anno, cell_calling_certainty_anno, cells):
        # Get unique cell types
        unique_cell_types = list(set(cells).union(set(adata.obs[cell_type_anno].unique())))
        # Initialize a new data array with zeros
        data = np.zeros((adata.shape[0], len(unique_cell_types)))
        # Loop over unique cell types
        for i, cell_type in enumerate(unique_cell_types):
            if cell_type in adata.obs[cell_type_anno].unique():
                # Get indices where cell type is the current cell type
                indices = np.where(adata.obs[cell_type_anno] == cell_type)[0]
                # Get corresponding certainties
                certainties = adata.obs.iloc[indices][cell_calling_certainty_anno].values
                # Assign these values to the corresponding column in data
                data[indices, i] = certainties
            
        # Create new AnnData object
        adata_new = ad.AnnData(X=data,
                                    obs=adata.obs,
                                    var=pd.DataFrame(index=unique_cell_types))
        return adata_new

    plot_adata = generate_anndata(adata_filt, cell_type_anno, cell_calling_certainty_anno, cells)
    
    # Check if all cell types in 'cells' are present in 'plot_adata'
    missing_cells = list(set(cells) - set(plot_adata.var_names))
    
    # Raise a warning if some cell types are missing
    if missing_cells:
        print(f"Warning: the following cell types are not found in the data and will be plotted as empty columns: {missing_cells}")
    
    # Create matrixplot
    sc.pl.matrixplot(plot_adata, var_names=cells, groupby=groupby, **kwargs)
