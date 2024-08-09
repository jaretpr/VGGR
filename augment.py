import torch
import torchvision.transforms.v2 as T
from torchvision.utils import save_image
from torchvision.transforms.functional import invert
from PIL import Image
import pandas as pd
from tqdm import tqdm

def augment():
    meta_path = './data/train/metadata/metadata.csv'
    augment_meta_path = './data/train/metadata/augmentation.csv'
    
    # Transformations
    jitter = T.ColorJitter(brightness=0.5, hue=0.3)
    five_crop = T.FiveCrop(size=((720, 1280)))
    to_tensor_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float, scale=True)])
    
    print('\nData Augmentation')
    augment_meta = pd.read_csv(augment_meta_path)
    with open(meta_path, 'a', encoding='utf-8') as meta_append:
        for i in tqdm(range(len(augment_meta)), 'File Progress'):    
            file = f"./data/train/img/{augment_meta.loc[i, 'folder']}/{augment_meta.loc[i, 'file']}"
            # Open Image
            img = Image.open(file)
            
            # Crop original images into five parts
            img = five_crop(img)
            # Turn images into tensors
            img = to_tensor_transform(img)
            top_left, top_right, bottom_left, bottom_right, center = img
            # Save all 5 parts as new images
            save_image(top_left, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_tl_{augment_meta.loc[i, 'file']}")
            save_image(top_right, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_tr_{augment_meta.loc[i, 'file']}")
            save_image(bottom_left, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_bl_{augment_meta.loc[i, 'file']}")
            save_image(bottom_right, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_br_{augment_meta.loc[i, 'file']}")
            save_image(center, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_cntr_{augment_meta.loc[i, 'file']}")
            # Append metadata file
            meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_tl_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
            meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_tr_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
            meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_bl_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
            meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_br_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
            meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_cntr_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
            
            # Open Image
            img = Image.open(file)
            # Invert and crop inverted images into five parts
            inv_img = invert(img)
            cropped_inv_img = five_crop(inv_img)
            inv_img = to_tensor_transform(inv_img)
            cropped_inv_img = to_tensor_transform(cropped_inv_img)
            i_top_left, i_top_right, i_bottom_left, i_bottom_right, i_center = cropped_inv_img
            # Save inverted images and append metadata file
            save_image(inv_img, f"./data/train/img/{augment_meta.loc[i, 'folder']}/inverted_{augment_meta.loc[i, 'file']}")
            meta_append.write(f"{augment_meta.loc[i, 'folder']},inverted_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
            # Save cropped inverted images
            save_image(i_top_left, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_tl_inverted_{augment_meta.loc[i, 'file']}")
            save_image(i_top_right, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_tr_inverted_{augment_meta.loc[i, 'file']}")
            save_image(i_bottom_left, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_bl_inverted_{augment_meta.loc[i, 'file']}")
            save_image(i_bottom_right, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_br_inverted_{augment_meta.loc[i, 'file']}")
            save_image(i_center, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_cntr_inverted_{augment_meta.loc[i, 'file']}")
            # Append metadata file
            meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_tl_inverted_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
            meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_tr_inverted_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
            meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_bl_inverted_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
            meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_br_inverted_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
            meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_cntr_inverted_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
            
            num = 1
            # Open Image
            img = Image.open(file)
            # Jitter
            jitter_imgs = [jitter(img) for _ in range(4)]
            # Save jittered images and append the metadata file
            for j_img in jitter_imgs:
                j_img = to_tensor_transform(j_img)
                save_image(j_img, f"./data/train/img/{augment_meta.loc[i, 'folder']}/jittered_{num}_{augment_meta.loc[i, 'file']}")
                meta_append.write(f"{augment_meta.loc[i, 'folder']},jittered_{num}_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
                # Crop jittered images into five parts
                cropped_j_img = five_crop(j_img)
                save_cropped_j_img = to_tensor_transform(cropped_j_img)
                j_top_left, j_top_right, j_bottom_left, j_bottom_right, j_center = save_cropped_j_img
                # Save all 5 parts as new images
                save_image(j_top_left, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_tl_jittered_{num}_{augment_meta.loc[i, 'file']}")
                save_image(j_top_right, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_tr_jittered_{num}_{augment_meta.loc[i, 'file']}")
                save_image(j_bottom_left, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_bl_jittered_{num}_{augment_meta.loc[i, 'file']}")
                save_image(j_bottom_right, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_br_jittered_{num}_{augment_meta.loc[i, 'file']}")
                save_image(j_center, f"./data/train/img/{augment_meta.loc[i, 'folder']}/crop_cntr_jittered_{num}_{augment_meta.loc[i, 'file']}")
                # Append metadata file
                meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_tl_jittered_{num}_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
                meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_tr_jittered_{num}_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
                meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_bl_jittered_{num}_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
                meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_br_jittered_{num}_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
                meta_append.write(f"{augment_meta.loc[i, 'folder']},crop_cntr_jittered_{num}_{augment_meta.loc[i, 'file']},{augment_meta.loc[i, 'genre']},{augment_meta.loc[i, 'game']},{augment_meta.loc[i, 'label key']}\n")
                
                num += 1
            
