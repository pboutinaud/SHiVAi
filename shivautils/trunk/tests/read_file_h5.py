import h5py
import pickle
import nibabel as nib
import ast
import numpy as np
import json

array_truth_vrs = {}
list_subject_id = ['SHARE0055', 'SHARE0228', 'SHARE0980', 'SHARE1008', 'SHARE1113', 'SHARE1322', 'SHARE1496', 'SHARE1560', 'SHARE1592', 'SHARE1598']
header_truth_vrs_initial = {}
header_truth_vrs_final = {}

# Ouvrir un fichier h5
with h5py.File('/bigdata/SHIVA/data/H5/mrishare/Unet_VRS_T1raw_40plus10.h5', 'r') as f:
    # Afficher les clés des groupes et des datasets dans le fichier
    #print("Keys: %s" % f.keys())
    
    # Accéder à un groupe spécifique
    groupe = f['10testing']
    
    # Accéder à des donnes sur un dataset spécifique dans le groupe
    donnes_on_dataset = groupe['images']
    #print(donnes_on_dataset.keys())

    # Crée un dict avec les header de toutes les images
    for key, val in donnes_on_dataset.attrs.items():
        #print("    %s: %s" % (key, val))
        header_truth_vrs_initial[key] = val
     
    # Lire les données du dataset
    dataset = donnes_on_dataset['images']

    donnees = dataset[:]
    for i in range(len(donnees)):
        array_truth_vrs[list_subject_id[i]] = donnees[i]


# Ici je récupère le header de l'image avec l'affine dedans en str, et je convertis cela en dict
# Car je veux reconstruire totalement les images des fichiers truth

# exemple pour un sujet
SHARE0055 = header_truth_vrs_initial['SHARE0055']
str_header = SHARE0055[0].decode('utf-8')
header_truth_vrs_0055 = eval(str_header, {'__builtins__': None}, {"np": np})

# conversion de tout les header en dict et mise dans un dict avec le nom des sujets en clés
for i in header_truth_vrs_initial:
    str_header = header_truth_vrs_initial[i][0].decode('utf-8')
    header_truth_vrs = eval(str_header, {'__builtins__': None}, {"np": np})
    header_truth_vrs_final[i] = header_truth_vrs


print(header_truth_vrs_initial['SHARE0055'])

#img = nib.load('/homes_unix/yrio/Documents/data/VRS/ISHARE/matlab_data/preproc_matlab/SHARE0055_T1_cropped/t1/SHARE0055_T1_cropped.nii.gz')
#header_arr = img.header.structarr
#header = img.header.copy()
#dict_header = dict(header)


# Ici je convertis le header de format dict au format <class 'nibabel.nifti1.Nifti1Header'>,
# exemple avec le sujet SHARE0055 :
header_original_SHARE0055 = nib.Nifti1Header()
for key, value in header_truth_vrs_final['SHARE0055'].items():
    setattr(header_original_SHARE0055, key, value)
#print(header_original_SHARE0055)
#img_modifié = nib.Nifti1Image(img.get_fdata(), img.affine, header_original_SHARE0055)
#print(img_modifié.header)

with open('/homes_unix/yrio/Documents/data/VRS/ISHARE/matlab_data/preproc_matlab/SHARE0055_T1_cropped/t1/SHARE0055_T1_cropped.nii.gz', 'wb') as file:
    array = file.get_fdata()
    affine = file.affine
    # Création d'une image Nifti avec l'array, l'affine et le header modifié
    nifti_image = nib.Nifti1Image(array, affine, header_original_SHARE0055)

    # Ecriture de l'image dans le fichier
    img_modifié = nib.Nifti1Image.to_file_map({'image': nifti_image}, file)
    print(img_modifié.header)




# Je récupère tout les header des 10 images du testset contenu dans le fichier '.h5', j'y extraie la matrice
# affine et je reconstitue les 10 images nibabel que je sauvegarde ensuite


#for key_array, array in array_truth_vrs.items():
    #for key_header, header in header_truth_vrs_final.items():
        #if key_array == key_header:





