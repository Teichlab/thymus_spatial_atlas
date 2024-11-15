#!/usr/bin/env python
# coding: utf-8

# # IBEXtractor2D 
# # 2D IBEX anaysis pipeline for regional and single cell data

# ### Last edit 18/03/2023

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL.TiffTags import TAGS
import os
from PIL import Image
import skimage




def read_dapi(path):
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    ppm = im.info['resolution'][0]
    return np.array(im), ppm




def cellpose_segmentation(
    im,
    ppm,
    model_type='cyto',
    gpu=True,
    diam_microns = 8,
    min_area_microns = 50,
    
):
    from cellpose import plot
    from cellpose import models 
    from cellpose import io
    from skimage import morphology

     # load models and load nuclear image to memory
    model = models.Cellpose(model_type=model_type,gpu=gpu)
    # run segmentation model
    print('starting cellpose segmentation')
    masks, flows, styles, diams = model.eval(
        x=im,
        diameter=int(ppm*diam_microns),
        resample=True,
        batch_size=1,
        flow_threshold=10,
        min_size=int(ppm*min_area_microns),
        cellprob_threshold=-1.5
    )
    print('removing small masks')
    masks = morphology.remove_small_objects(masks,2*min_area_microns*ppm)
    labels = np.unique(masks)
    print('found '+str(len(labels.astype(int)))+' segmentation labels')
    return masks


def seg_to_sq(
    im,
    masks,
    show=True,
):
    import squidpy as sq
    img = sq.im.ImageContainer(library_id=['processed_image'])
    img.add_img(im,layer = 'nuclear')
    # add segmentation and set as segmentation level
    img.add_img(masks,layer = 'segmented_cellpose')
    img['segmented_cellpose'].attrs["segmentation"] = True
    if show:
        img.show("nuclear",segmentation_layer='segmented_cellpose',segmentation_alpha=0.4,dpi=500)
    return img
    
def perim_extract(
    img, # image container
    ppm,
    inside = 1,
    outside = 1
):
    import scipy
    import skimage
    import squidpy as sq
    def perimiter_segmentation(
        LabelImage,
        inside = inside,
        outside= outside
    ):
        eroded = LabelImage * (np.abs(scipy.ndimage.laplace(LabelImage)) > 0)
        for i in range(inside-1):
            eroded = LabelImage * (np.abs(scipy.ndimage.laplace(eroded)) > 0)
            labels = skimage.segmentation.expand_labels(LabelImage, distance=outside)-(LabelImage-eroded)
        return scipy.ndimage.median_filter(labels,size=2)

    sq.im.segment(
        img=img,
        channel=0,
        layer = "segmented_cellpose",
        method=perimiter_segmentation,
        layer_added='perimiter_segmentation',
        inside = int(inside*ppm), # gap from eroded 
        outside = int(outside*ppm)
    )
    
    return img 

def extract_morpho_features(
    im_cell,
    im, 
):
    
    def MeasureTextureFeatures(regionmask,intensity_image): # adated from squidpy sourceode 
        a = intensity_image.copy()
        a[regionmask] = 0
        b = intensity_image.copy()
        b = b-a 
        angles = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
        props = ("contrast", "dissimilarity", "homogeneity", "correlation", "ASM")
        distances = (1,)
        features = {}
        comatrix = graycomatrix(b.astype(np.uint8), distances=distances, angles=angles, levels=256)
        for p in props:
            tmp_features = graycoprops(comatrix, prop=p)
            for d_idx, dist in enumerate(distances):
                for a_idx, a in enumerate(angles):
                    features[f"{'texture'}_{p}_dist-{dist}_angle-{a:.2f}"] = tmp_features[d_idx, a_idx]
        return features
    
    from skimage import data, util
    from skimage.measure import label, regionprops
    from skimage.feature import graycomatrix, graycoprops

    props = ('label','centroid','area','perimeter')
    NuclearData = skimage.measure.regionprops_table(
        label_image=im_cell,
        intensity_image=im,
        properties=props, 
        cache=True,
        separator='-',
        extra_properties=[MeasureTextureFeatures]
    )
    # Arrange data in df and caculate cell circularity
    import math 
    dfseg_nuc = pd.DataFrame(data=NuclearData)
    circularity = (4 * math.pi * dfseg_nuc['area']) / (dfseg_nuc['perimeter'] * dfseg_nuc['perimeter'])
    dfseg_nuc['circularity'] = circularity
    dfseg_nuc.replace([np.inf, -np.inf, None], 0, inplace=True)
    
    import warnings
    warnings.filterwarnings("ignore")
    TextFeatures = pd.DataFrame.from_dict(dfseg_nuc['MeasureTextureFeatures'].values[0], orient='index').T
    for row in dfseg_nuc['MeasureTextureFeatures'].values[1::]:
        TextFeatures1 = pd.DataFrame.from_dict(row, orient='index').T
        TextFeatures = TextFeatures.append(TextFeatures1,ignore_index=True)
    dfseg_nuc = pd.concat([dfseg_nuc, TextFeatures],axis=1)
    dfseg_nuc= dfseg_nuc.drop(columns=['MeasureTextureFeatures'])

    return dfseg_nuc



def channel_measurments(LabelImage,IntImage,marker):
    props = ('mean_intensity','max_intensity','area')
    ImageDatatmp = skimage.measure.regionprops_table(LabelImage, intensity_image=IntImage, properties=props, cache=True, separator='-')
    k_new = 'ch_'+marker+'_total_int'
    ImageDatatmp['area'] = ImageDatatmp['mean_intensity']*ImageDatatmp['area'] # calculate total int and change name
    ImageDatatmp[k_new] = ImageDatatmp.pop('area')
    k_new = 'ch_'+marker+'_mean_int'
    ImageDatatmp[k_new] = ImageDatatmp.pop('mean_intensity') # add channel label to mean int
    k_new = 'ch_'+marker+'_max_int'
    ImageDatatmp[k_new] = ImageDatatmp.pop('max_intensity') # add channel label to max int
    dfsegTemp = pd.DataFrame(data=ImageDatatmp)
    return dfsegTemp


def radius_mean(radius,ppm,data ,image):
    print('radius: '+str(int(radius/ppm)))
    tempdict = {}
    for idx,c in enumerate(np.column_stack((data['centroid-1'].values,data['centroid-0'].values))):
        image[int(c[1]),int(c[0])]
        disk = skimage.draw.disk([int(c[1]),int(c[0])],radius,shape=image.shape)
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
    perif_r_microns=None,

):
    for idx,marker in enumerate(meta_data['protein']): # add channel date for entire cell and add channel data for perimiter
        if not(meta_data['exclude'][idx]):
            print(marker)
            im = Image.open(series_path+meta_data['file'][idx])
            print(series_path+meta_data['file'][idx])
            print('mean before subtract - '+str(np.mean(im)))
            im = np.array(im).astype('float64') - meta_data['threshold'][idx] # subtract background
            im[np.where(im<0)] = 0  
            im = im.astype('uint8')
            print('mean after subtract - '+str(np.mean(im)))
            dfsegTemp = channel_measurments(im_cell,im,marker)
            dfseg_nuc = pd.concat([dfseg_nuc, dfsegTemp], axis=1)
            # add perimiter segmentation
            if add_perim:
                dfsegTemp_prim = channel_measurments(im_perim,im,'perim_'+marker)
                dfseg_nuc = pd.concat([dfseg_nuc, dfsegTemp_prim], axis=1)
            # add radius segmentations  
            if add_radius:
                for r in perif_r_microns:
                    tempdict = radius_mean(radius=int(r*ppm),ppm=ppm,data = dfseg_nuc,image=im)
                    dfseg_nuc['ch_'+marker+ '_r'+str(int(r))] = dfseg_nuc.index.map(tempdict)

        else:
            print('skipping - '+marker)
            
    return dfseg_nuc

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
        # anno_orig = skimage.transform.resize(anno,dim,preserve_range=True).astype('uint8') 
        anno_orig = anno

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

    
def anno_to_cells(
    df_cells,
    df_morphology,
    numerical_annotations, 
    categorical_annotation_names, 
    categorical_annotation_number_names, 
):
    
    print('make sure the coordinate systems are alligned e.g. axes are not flipped') 
    a = np.vstack([df_morphology['x'],df_morphology['y']])
    b = np.vstack([df_cells['centroid-1'],df_cells['centroid-0']])
    # xi = np.vstack([dfseg['centroid-1'],dfseg['centroid-0']]).T
    plt.figure(dpi=100, figsize=[10,10])
    plt.title('cell space')
    plt.plot(b[0],b[1],'.', markersize=1)
    plt.show()
    plt.figure(dpi=100, figsize=[10,10])
    plt.plot(a[0],a[1],'.', markersize=1)
    plt.title('morpho spcae')
    plt.show()

    import scipy
    # migrate continues annotations
    xi = np.vstack([df_cells['centroid-1'],df_cells['centroid-0']]).T
    for k in numerical_annotations:
        print('migrating - '+k+' to segmentations')
        df_cells[k] = scipy.interpolate.griddata(points=a.T, values = df_morphology[k], xi=b.T,method='nearest')
        # plt.title(k)
        # df_cells[k].hist(bins=5)
        # plt.show()
    if categorical_annotation_names:
        # migrate categorial annotations
        for idx,k in enumerate(categorical_annotation_names):
            df_cells[categorical_annotation_number_names[idx]] = scipy.interpolate.griddata(points=a.T, values = df_morphology[categorical_annotation_number_names[idx]], xi=b.T,method='nearest')
            dict_temp = dict(zip(df_morphology[categorical_annotation_number_names[idx]].value_counts().keys(),df_morphology[k].value_counts().keys()))
            print('migrating - '+k+' to segmentations')
            df_cells[k] = df_cells[categorical_annotation_number_names[idx]].map(dict_temp)
            print(df_cells[k].value_counts())
        
    return df_cells



# def ibex_csv_to_anndata(df, channelkeys = vars ,obskeys = obs_anno, spatialkeys = xylist,dapiImage,metadata)   
# )

# def correct_and_trasnform(
# ) 


def ibextractor_feature_pipeline(
    path,
    meta_data,
    sample_name,
    im, 
    ppm,
    expected_diam, 
    perif_r = [10.5,25.5],
    
):

    # cellpose segmentation
    masks = cellpose_segmentation(im,ppm,diam_microns=expected_diam,min_area_microns=expected_diam*5)
    masksP = Image.fromarray(masks)
    masksP.save(path+sample_name+'_masks.tif')

    # transfer to squidpy
    img = seg_to_sq(im, masks)

    # create perimiter masks per cell 2um wide 
    img = perim_extract(img,ppm)
    # plot new levels
    plt.rcParams['figure.dpi'] = 500
    fig, axes = plt.subplots(1, 3)
    img.show("nuclear", ax=axes[0])
    _ = axes[0].set_title("nuclear")
    img.show("nuclear", ax=axes[1],segmentation_layer='segmented_cellpose',segmentation_alpha=0.7)
    _ = axes[1].set_title("nuclear+seg")
    img.show("nuclear", ax=axes[2],segmentation_layer='perimiter_segmentation',segmentation_alpha=0.7)
    _ = axes[2].set_title("Perimiter")
    fig.tight_layout()
    # fig.savefig('figures/SegCropExample.png')

    # save segmentation images 
    from PIL import Image
    im_cell = np.squeeze(np.array(img['segmented_cellpose'].compute()))
    im_cellP = Image.fromarray(im_cell)
    im_cellP.save(path+'Sample_05_seg.tif',format='tiff')
    # save segmentation perim
    im_perim = np.squeeze(np.array(img['perimiter_segmentation'].compute()))
    im_perimP = Image.fromarray(im_perim)
    im_perimP.save(path+'Sample_05_seg_perim.tif',format='tiff')

    #  this step is very long and expected to take about to 40min per 100K labels
    dfseg_nuc = extract_morpho_features(im_cell,im)
    dfseg_nuc.to_csv(path+'FullNuclearFeatures.csv')

    # add individual channel data 
    dfseg_nuc = add_channel_measurments(
        series_path=path+'series/',
        dfseg_nuc=dfseg_nuc,
        meta_data=meta_data,
        ppm=ppm,
        im_cell=im_cell,
        add_perim=True,
        im_perim=im_perim,
        add_radius=True,
        perif_r_microns=perif_r,
    )

    dfseg_nuc.to_csv(path+'full_channel_segmentation.csv')