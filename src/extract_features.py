import pandas as pd
import feature_hair, feature_pen_marks, feature_A, feature_B, feature_C # all of our features
import cv2
import pandas as pd
from skimage.feature import hog
import time

###################################
# CHANGE THIS SECTION TO OWN PATH 
data_path = "../data/"            
img_path = data_path + "imgs/"    
mask_path = data_path + "masks/"  
###################################

class Picture:
    def __init__(self, input_ID: str, img_path = img_path, mask_path = mask_path) -> None:
        self.input_ID = input_ID
        self.img_file = img_path + self.input_ID
        self.mask_file = (mask_path + self.input_ID).replace(".png","_mask.png")
        self.mask_img = cv2.imread(self.mask_file, cv2.IMREAD_GRAYSCALE)
        self.mask_img = cv2.resize(self.mask_img, (64,64))
        self.mask_img = self.mask_img > 0
        self.img_org = cv2.imread(self.img_file)
        self.img_org = cv2.resize(self.img_org,(64,64))
        self.img_visual = cv2.cvtColor(self.img_org, cv2.COLOR_BGR2RGB)
        self._grey = cv2.cvtColor(self.img_org, cv2.COLOR_RGB2GRAY)  
        self._img_grey_resized = cv2.resize(self._grey, (64, 64))

    def mask(self) -> list:
        return self.mask_file

    def clean_picture(self) -> list:
        
        # Step 1 call feature_hair (need greyscale)
        self.blackhat, self.hair_mask, self.hairless = feature_hair.removeHair(self.img_org, self._grey)
        # Step 2 call feature_pen (on hair removed pic)
        self.clean_image, self.pen_mask = feature_pen_marks.remove_pen_marks(self.hairless, self.mask_img)
        return self.clean_image
    
    def hog_feature(self):
        feature = hog(
            self._img_grey_resized, orientations=9, 
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2)
        )
        return feature
# ------------------
if __name__ == "__main__":
    start = time.time()
    cancerous = {"BCC","MEL","SCC"}
    rows = []
    hog_rows = []

##############################################################################
    df = pd.read_csv("../metadata_with_group.csv") # CHANGE THIS TO OWN PATH 
##############################################################################

    df = df.drop(columns=['Unnamed: 0', 'patient_id', 'lesion_id', 'smoke', 'drink',
       'background_father', 'background_mother', 'age', 'pesticide', 'gender',
       'skin_cancer_history', 'cancer_history', 'has_piped_water',
       'has_sewage_system', 'fitspatrick', 'region', 'diameter_1',
       'diameter_2','itch', 'grew', 'hurt', 'changed', 'bleed',
       'elevation','biopsed', 'group_id'])
    df = df.head(500) # CHANGE THIS TO THE DESIRED NUMBER OF PICTURES OR COMMENT IT OUT FOR THE WHOLE DATA SET! 

    for _, row in df.iterrows():
        img = Picture(row["img_id"])
        img_clean = img.clean_picture()

        rows.append({
            "ID": img.input_ID,
            "Asymmetry": feature_A.get_asymmetry(img.mask_img),
            "Border": feature_B.compactness_score(img.mask_img),            
            "HSV_Saturation_Variance": feature_C.hsv_var(img_clean,img.mask_img)[1],
            "Cancerous": 1 if row["diagnostic"] in cancerous else 0
        })
        hog_rows.append(img.hog_feature())


    features = pd.DataFrame(rows)
    hog_features = pd.DataFrame(hog_rows)
    combined_df = pd.concat([features.reset_index(drop=True), hog_features.reset_index(drop=True)], axis=1)

    combined_df.to_csv("../data/features.csv") # FEATURE EXTRACTION PATH, CHANGE IF DESIRED! 
    print("Feature extraction done!")
