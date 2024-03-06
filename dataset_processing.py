import os
import random
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import imutils


class image_processing:

    def __init__(self) -> None:

        pass

    def augmented_data(self,file_dir: os.path, n_generated_samples:int, save_to_dir:os.path) -> None:

        self.data_gen = ImageDataGenerator(rotation_range=10,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      shear_range=0.1,
                                      brightness_range=(0.3, 1.0),
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      fill_mode='nearest')
        print("data augmenting --------------------------------")
        print("data augmenting --------------------------------")
        for filename in os.listdir(file_dir):
            image = cv2.imread(file_dir + '/' + filename)
            image = image.reshape((1,) + image.shape)
            save_prefix = 'aug_' + filename[:-4]
            i = 0
            for batch in self.data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir, save_prefix=save_prefix,
                                       save_format="jpg"):
                i += 1
                if i > n_generated_samples:
                    break




    def crop_image(self, image:np.ndarray, plot:bool = False) -> np.ndarray:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thres = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thres = cv2.erode(thres, None, iterations=2)
        thres = cv2.dilate(thres, None, iterations=2)

        cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        brightness_range = (0.3, 1.0),
        c = max(cnts, key=cv2.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

        print("croping -------------------------")
        print("croping -------------------------")

        if plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(image)

            plt.tick_params(axis='both', which='both',
                            top=False, bottom=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

            plt.title('Original Image')

            plt.subplot(1, 2, 2)
            plt.imshow(new_image)

            plt.tick_params(axis='both', which='both',
                            top=False, bottom=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

            plt.title('Cropped Image')
            plt.show()

        return new_image

    def crop_aug_brain_tumor_in_folder(self, folder1:str,  folder2:str,plot = False) -> None:

        for filename in os.listdir(folder1):

            img = cv2.imread(folder1 + filename)
            img = self.crop_image(img, plot)
            cv2.imwrite(folder1 + filename, img)
        for filename in os.listdir(folder2):
            img = cv2.imread(folder2 + filename)
            img = self.crop_image(img, plot)
            cv2.imwrite(folder2 + filename, img)

        print("croping -------------------------")
        print("croping -------------------------")



    def split_data(self,data_path:os.path, output_path: os.path, train_ratio: float =0.85, val_ratio: float=0.15) -> None:

        os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists or create it
        self.train_folder = os.path.join(output_path, 'train')  # declare the  Path for the  training data folder
        self.val_folder = os.path.join(output_path, 'val')  # Path for the validation data folder
        os.makedirs(self.train_folder, exist_ok=True)  # Create the training data folder if it doesn't exist
        os.makedirs(self.val_folder, exist_ok=True)  # Create the validation data folder if it doesn't exist
        categories = os.listdir(data_path)

        for category in categories:

            self.category_folder = os.path.join(data_path, category)  # Path to the category folders in the source data

            valid_image_files = []
            for filename in os.listdir(self.category_folder):  # list all the a in the Brain_tumor filders
                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):

                    valid_image_files.append(filename)

            random.shuffle(valid_image_files)

            num_images = len(valid_image_files)  # check the numbers of valid images we have
            num_train = int(num_images * train_ratio)
            num_val = int(num_images * val_ratio)
            train_category_folder = os.path.join(self.train_folder,
                                                 category)  # declear Path for the category folders in the training data
            val_category_folder = os.path.join(self.val_folder, category)  # Path for the category in the validation data

            os.makedirs(train_category_folder,
                        exist_ok=True)  # Create the category folder in training if it doesn't exist
            os.makedirs(val_category_folder,
                        exist_ok=True)  # Create the category folder in validation if it doesn't exist
            train_data = valid_image_files[
                         :num_train]  # starting from 0 to the amount of images num, Select the first num_train images for training
            val_data = valid_image_files[num_train:num_train + num_val]  # Select the next num_val images for validation
            for filename in train_data:

                src_path = os.path.join(self.category_folder, filename)  # Source path for the images in the category folder
                dest_path = os.path.join(train_category_folder,
                                         filename)  # Destination path for the images in the training category folder
                shutil.copy(src_path, dest_path)  # Copy the images from source to destinatio

            for filename in val_data:

                    src_path = os.path.join(self.category_folder,
                                            filename)  # Source path for the images in the category folder
                    dest_path = os.path.join(val_category_folder,
                                             filename)  # Destination path for the images in the validation category folder
                    shutil.copy(src_path, dest_path)  # Copy the images from source to destination





if __name__ == '__main__':

    image_process = image_processing()

    aug = False
    split = True
    crop = True

    if aug == True:

        augmented_data_path = 'augmented_data/'
        Brain_tumor_path = 'yes'
        No_Brain_tumor_path = 'no'

        image_process.augmented_data(file_dir=Brain_tumor_path, n_generated_samples=6, save_to_dir=augmented_data_path + 'yes')
        image_process.augmented_data(file_dir=No_Brain_tumor_path, n_generated_samples=9, save_to_dir=augmented_data_path + 'no')

    if crop == True:
        folder1 = "augmented_data/yes/"
        folder2 = "augmented_data/no/"

        image_process.crop_aug_brain_tumor_in_folder(folder1=folder1, folder2=folder2)


    if split == True:

        data_path = "augmented_data"  # path to the source Brain_tumor folder
        output_path = "dataset"  # Path where the split data will be saved
        image_process.split_data(data_path,output_path)



