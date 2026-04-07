import pandas as pd
import matplotlib.pyplot as plt
import feature_hair, feature_pen_marks, feature_A, feature_B, feature_C # all of our features
import cv2
import pandas as pd
import time
import concurrent.futures

# ------------------
data_path = "../data/"
img_path = data_path + "imgs/"
mask_path = data_path + "masks/"

class Picture:
    def __init__(self, input_ID: str, img_path = img_path, mask_path = mask_path) -> None:
        self.input_ID = input_ID
        self.img_file = img_path + self.input_ID
        self.mask_file = (mask_path + self.input_ID).replace(".png","_mask.png")
        self.mask_img = cv2.imread(self.mask_file, cv2.IMREAD_GRAYSCALE)
        self.mask_img = self.mask_img > 0
        self.img_org = cv2.imread(self.img_file)
        self.img_org = cv2.cvtColor(self.img_org, cv2.COLOR_BGR2RGB)  # convert to RGB for matplotlib
        self._grey = cv2.cvtColor(self.img_org, cv2.COLOR_RGB2GRAY)  #img_file

    def mask(self) -> list:
        return self.mask_file

    def clean_picture(self) -> list:
        
        # Step 1 call feature_hair (need greyscale)
        self.blackhat, self.hair_mask, self.hairless = feature_hair.removeHair(self.img_org, self._grey)
        # Step 2 call feature_pen (on hair removed pic)
        
        return 
# ------------------
def process_single_image(row_data):
    """This is the 'worker' function. One CPU core will run this for one image."""
    img_id = row_data["img_id"]
    diagnostic = row_data["diagnostic"]
    
    img = Picture(img_id)
    
    # Calculate features
    feature_dict = {
        "ID": img.input_ID,
        "Asymmetry": feature_A.get_asymmetry(img.mask_img),
        "Border": feature_B.get_compactness(img.mask_img),
        "[Color placeholder]": "placeholder",
        "Cancerous": 1 if diagnostic in {"BCC", "MEL", "SCC"} else 0
    }
    return feature_dict

if __name__ == "__main__":
    start = time.time()
    cancerous = {"BCC","MEL","SCC"}
    rows = []
    patient_IDs = []

    # Getting all the unique IMG IDs 
    df = pd.read_csv("adjacent_metadata_small.csv")

    for _, row in df.iterrows():
        img = Picture(row["img_id"])
            
        rows.append({
            "ID": img.input_ID,
            "Asymmetry": feature_A.get_asymmetry(img.mask_img),
            "Border": feature_B.get_compactness(img.mask_img),
            "[Color placeholder]": "placeholder",
        "Cancerous": 1 if row["diagnostic"] in cancerous else 0
        })

    #tasks = df.to_dict('records')
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #    # The executor maps the worker function to every task in the list simultaneously
    #    results_list = list(executor.map(process_single_image, tasks))

    features = pd.DataFrame(rows)
    #features = pd.DataFrame(results_list)
    end = time.time()
    print(features)
    print(f"Time it took: {end-start}s")
