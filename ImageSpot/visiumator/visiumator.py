#!/usr/bin/env python
# coding: utf-8

# # visiumator 
# # visium anaysis pipeline for regional and transcriptomics

# ### Last edit 18/03/2023

from PIL.TiffTags import TAGS
import skimage
# import openslide
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import glob
import numpy as np
import os, sys, re
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import listdir
from os.path import isfile, join
from skimage.transform import rescale, resize, downscale_local_mean,  warp, AffineTransform
from PIL import Image, ImageDraw, ImageFilter
from skimage import measure, img_as_ubyte, color 
from skimage.measure import ransac
import pandas as pd
import math 
from sklearn.metrics.pairwise import euclidean_distances
import cv2
from cellpose import utils, io, models, plot
from numpy import genfromtxt
import json
import scanpy as sc
import anndata as ad
import shutil


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # to show output from all the lines in a cells
pd.set_option('display.max_column',None) # display all the columns in pandas
pd.options.display.max_rows = 100
from datetime import date
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

        
def downsample_image(img,width,height,target_max_size=5000,greyscale=False):
    # downsample Image
    scalefactor = max([width,height])/target_max_size # Scale image down to analysis_image_size widht or hight pixels 
    dim = (int(width * (1/scalefactor)),int(height * (1/scalefactor)))
    img_rescaled = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
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
    img = io.imread(path+'imagespot_preprocessing/spatial/tissue_lowres_image.png')
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
        img5k = io.imread(path+'imagespot_preprocessing/spatial/tissue_hires5K_image.png')
        sample = list(AnnVis.uns['spatial'].keys())
        AnnVis.uns['spatial'][sample[0]]['images']['hires5K'] = img5k
        AnnVis.write_h5ad(path+'imagespot_preprocessing/corrected_raw_adata.h5ad')

    
    return BarcodeTo_XY




def generate_hires_grid(
    im,
    spot_diameter,
    pixels_per_micron,
):
    import skimage
    helper = spot_diameter*pixels_per_micron
    X1 = np.linspace(helper,im.shape[0]-helper,round(im.shape[0]/helper))
    Y1 = np.linspace(helper,im.shape[1]-2*helper,round(im.shape[1]/(2*helper)))
    X2 = X1 + spot_diameter/2*pixels_per_micron
    Y2 = Y1 + helper
    Gx1, Gy1 = np.meshgrid(X1,Y1)
    Gx2, Gy2 = np.meshgrid(X2,Y2)
    positions1 = np.vstack([Gy1.ravel(), Gx1.ravel()])
    positions2 = np.vstack([Gy2.ravel(), Gx2.ravel()])
    positions = np.hstack([positions1,positions2])
    
    return positions


def grid_anno(
    im,
    annotation_image_list,
    annotation_image_names,
    annotation_label_list,
    spot_diameter,
    pixels_per_micron,
    
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
        positions 
            grid positions

    """
    print('generating grid with spot size - '+str(spot_diameter)+', with resolution of - '+str(pixels_per_micron)+' ppm')
    positions = generate_hires_grid(im,spot_diameter,pixels_per_micron)
    positions = positions.astype('float32')
    print(positions)
    dim = [im.shape[0],im.shape[1]]
    # transform tissue annotation images to original size
    from tqdm import tqdm
    from scipy.spatial import distance
    radius = spot_diameter/2
    df = pd.DataFrame(
        np.vstack((np.array(range(len(positions.T[:,0]))),positions.T[:,0],
                   positions.T[:,1])).T,
        columns=['index','x','y'])
    for idx0,anno in enumerate(annotation_image_list):
        anno_orig = cv2.resize(anno, im.shape[:2], interpolation = cv2.INTER_AREA)        #
        # anno_orig = anno

        anno_dict = {}
        number_dict = {}
        name = f'{anno=}'.split('=')[0]
        print(annotation_image_names[idx0])
        for idx1,pointcell in tqdm(enumerate(positions.T)):
            # disk = skimage.draw.disk([int(pointcell[1]),int(pointcell[0])],radius)
            disk = skimage.draw.disk([int(pointcell[1]/pixels_per_micron),int(pointcell[0]/pixels_per_micron)],int(radius/pixels_per_micron))

            anno_dict[idx1] = annotation_label_list[idx0][int(np.median(anno_orig[disk]))]
            number_dict[idx1] = int(np.median(anno_orig[disk]))
        df[annotation_image_names[idx0]] = anno_dict.values()
        df[annotation_image_names[idx0]+'_number'] = number_dict.values()
    df['index'] = df['index'].astype(int)
    df.set_index('index', drop=True, inplace=True)
    return df



# measure for each spot the mean closest distance to the x closest spoint of a given structure, resolution is x 
def dist2cluster(
    df,
    annotation,
    distM,
    resolution=4,
    calc_dist=True,
    logscale = False
):
    Dist2ClusterAll = {}    
    categories = np.unique(df[annotation])
    for idx, c in enumerate(categories): # measure edistange to all
        indextmp = df[annotation]==c
        if len(np.where(indextmp)[0])>resolution: # this is an important point to see if there are enough points from a structure to calculate mean distance 
            Dist2ClusterAll[c] =  np.median(np.sort(distM[indextmp,:],axis=0)[range(resolution),:],axis=0) # was 12

    # update annotations in AnnData 
    for c in categories: 
        indextmp = df[annotation]==c
        if len(np.where(indextmp)[0])>resolution: # this is an important point to see if there are enough points from a structure to calculate mean distance 
            if calc_dist:
                if logscale:
                    df["L2_dist_log10_"+annotation+'_'+c] = np.log10(Dist2ClusterAll[c])
                else:
                    df["L2_dist_"+annotation+'_'+c] = Dist2ClusterAll[c]
            df[annotation] = categories[np.argmin(np.array(list(Dist2ClusterAll.values())),axis=0)]
    return Dist2ClusterAll


def axis_2p_norm(
    Dist2ClusterAll,
    structure_list,
    weights = [1,1], 
):
    import warnings
    warnings.filterwarnings("ignore")
    # CMA calculations 
    fa = weights[0]
    fb = weights[0]
    axis = np.array([( fa*int(a) - fb*int(b) ) / ( fa*int(a) + fb*int(b) ) for a,b in zip(Dist2ClusterAll[structure_list[0]], Dist2ClusterAll[structure_list[1]])])
    return axis

def bin_axis(
    ct_order,
    cutoff_vales,
    df,
    axis_anno_name,
):

    # # manual annotations
    df['manual_bin_'+axis_anno_name] = 'unassigned'
    df['manual_bin_'+axis_anno_name] = df['manual_bin_'+axis_anno_name].astype('object')
    df.loc[np.array(df[axis_anno_name]<cutoff_vales[0]) ,'manual_bin_'+axis_anno_name] = ct_order[0]
    for idx,r in enumerate(cutoff_vales[:-1]):
        print(ct_order[idx+1])
        print(str(cutoff_vales[idx])+','+str(cutoff_vales[idx+1]))
        df.loc[np.array(df[axis_anno_name]>=cutoff_vales[idx]) & np.array(df[axis_anno_name]<cutoff_vales[idx+1]),'manual_bin_'+axis_anno_name] = ct_order[idx+1]

    df.loc[np.array(df[axis_anno_name]>=cutoff_vales[-1]),'manual_bin_'+axis_anno_name] = ct_order[-1]
    df['manual_bin_'+axis_anno_name] = df['manual_bin_'+axis_anno_name].astype('category')
    df['manual_bin_'+axis_anno_name+'_int'] =  df['manual_bin_'+axis_anno_name].cat.codes
   
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
