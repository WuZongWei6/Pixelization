import os
from data.base_dataset import BaseDataset, get_transform,get_params
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        if opt.phase == 'train':
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
            self.opt = opt
            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
            self.A_size = len(self.A_paths)  # get the size of dataset A
            self.B_size = len(self.B_paths)  # get the size of dataset B
            btoA = self.opt.direction == 'BtoA'
            input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
            output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
            self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
            self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
            self.factor_mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}
        elif opt.phase == 'test':
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
            self.opt = opt
            self.A_paths = sorted(
                make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
            self.B_paths = sorted(
                make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
            self.A_size = len(self.A_paths)  # get the size of dataset A
            self.B_size = len(self.B_paths)  # get the size of dataset B
            btoA = self.opt.direction == 'BtoA'
            self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
            self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc
            self.factor_mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self.opt.phase == 'train':
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            factor = os.path.basename(B_path).split('_')[0]
            label = self.factor_mapping[int(factor)]
            A_img = Image.open(A_path).convert('RGB')
            A_img = A_img.resize((480,480), Image.BICUBIC)
            x1 = 90#random.randint(0,480-300)
            y1 = 90#random.randint(0,480-300)
            box=(x1,y1,x1+300,y1+300)
            A_img = A_img.crop(box)
            A_img = A_img.resize((480,480), Image.BICUBIC)

            if label==5:
                A_image = A_img.resize((476,476), Image.BICUBIC)
            #if label==6:
                #A_image = A_img.resize((560,560), Image.BICUBIC)
            B_img = Image.open(B_path).convert('RGB')
            # 灰度
            B_gray = B_img.convert('L')
            I_array = np.array(B_gray)  #
            img = np.expand_dims(I_array, axis=2)
            img = np.concatenate((img, img, img), axis=-1)
            B_gray = Image.fromarray(img)


            A = self.transform_A(A_img)
            B = self.transform_B(B_img)
            B_gray = self.transform_B(B_gray)

            return {'A': A, 'B': B, 'B_gray': B_gray, 'label': label, 'A_paths': A_path, 'B_paths': B_path}
        elif self.opt.phase == 'test':
            A_path = self.A_paths[index % self.A_size]
            base_name = os.path.basename(A_path)
            B_path = os.path.join(self.opt.dataroot, self.opt.phase + 'B', base_name)
            factor = os.path.basename(B_path).split('_')[0]
            label = self.factor_mapping[int(factor)]
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
            B_gray = B_img.convert('L')
            I_array = np.array(B_gray)  #
            img = np.expand_dims(I_array, axis=2)
            img = np.concatenate((img, img, img), axis=-1)
            B_gray = Image.fromarray(img)
            #A_img = A_img.resize((192,192),Image.BICUBIC)
            transform_params = get_params(self.opt)
            A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
            gray_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
            A = A_transform(A_img)
            B = B_transform(B_img)
            B_gray = gray_transform(B_gray)
            return {'A': A, 'B': B, 'B_gray': B_gray, 'label': label, 'A_paths': A_path, 'B_paths': B_path}

        #return {'A': A, 'B': B, 'B_gray': B_gray, 'label': label,'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)



#
# class UnalignedDataset(BaseDataset):
#     """
#     This dataset class can load unaligned/unpaired datasets.
#
#     It requires two directories to host training images from domain A '/path/to/data/trainA'
#     and from domain B '/path/to/data/trainB' respectively.
#     You can train the model with the dataset flag '--dataroot /path/to/data'.
#     Similarly, you need to prepare two directories:
#     '/path/to/data/testA' and '/path/to/data/testB' during test time.
#     """
#
#     def __init__(self, opt):
#         """Initialize this dataset class.
#
#         Parameters:
#             opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseDataset.__init__(self, opt)
#         self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
#         self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
#
#         self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
#         self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
#         self.A_size = len(self.A_paths)  # get the size of dataset A
#         self.B_size = len(self.B_paths)  # get the size of dataset B
#         btoA = self.opt.direction == 'BtoA'
#         input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
#         output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
#         self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
#         self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
#         self.factor_mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}
#     def __getitem__(self, index):
#         """Return a data point and its metadata information.
#
#         Parameters:
#             index (int)      -- a random integer for data indexing
#
#         Returns a dictionary that contains A, B, A_paths and B_paths
#             A (tensor)       -- an image in the input domain
#             B (tensor)       -- its corresponding image in the target domain
#             A_paths (str)    -- image paths
#             B_paths (str)    -- image paths
#         """
#         A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
#         # if self.opt.serial_batches:   # make sure index is within then range
#         #     index_B = index % self.B_size
#         # else:   # randomize the index for domain B to avoid fixed pairs.
#         #     index_B = random.randint(0, self.B_size - 1)
#         index_B = random.randint(0, self.B_size - 1)
#         B_path = self.B_paths[index_B]
#         factor = os.path.basename(B_path).split('_')[0]
#         label = self.factor_mapping[int(factor)]
#         A_img = Image.open(A_path).convert('RGB')
#         B_img = Image.open(B_path).convert('RGB')
#         # 灰度
#         B_gray = B_img.convert('L')
#         I_array = np.array(B_gray)  #
#         img = np.expand_dims(I_array, axis=2)
#         img = np.concatenate((img, img, img), axis=-1)
#         B_gray = Image.fromarray(img)
#
#
#
#         # 把A(clipart)转成32*32,48*48,64*64
#         # A_img_32 = A_img.resize((32, 32), Image.NEAREST)
#         # A_img_32 = A_img_32.resize((256, 256), Image.NEAREST)
#         # A_img_48 = A_img.resize((48, 48), Image.NEAREST)
#         # A_img_48 = A_img_48.resize((256, 256), Image.NEAREST)
#         # A_img_64 = A_img.resize((64, 64), Image.NEAREST)
#         # A_img_64 = A_img_64.resize((256, 256), Image.NEAREST)
#         # apply image transformation
#         A = self.transform_A(A_img)
#         B = self.transform_B(B_img)
#         B_gray = self.transform_B(B_gray)
#         # A_32 = self.transform_A(A_img_32)
#         # A_48 = self.transform_A(A_img_48)
#         # A_64 = self.transform_A(A_img_64)
#         # ------------------------------------------------------
#         # import torchvision.transforms as transforms
#         # import matplotlib.pyplot as plt
#         # unloader = transforms.ToPILImage()
#         # blur_rgb1_show = unloader(A_48)
#         # plt.subplot(1, 2, 1)
#         # plt.imshow(blur_rgb1_show)
#         # plt.axis('on')  # 关掉坐标轴为 off
#         # plt.title('image_nearest')  # 图像题目
#         # blur_rgb2_show = A_img_48
#         # plt.subplot(1, 2, 2)
#         # plt.imshow(blur_rgb2_show)
#         # plt.axis('on')  # 关掉坐标轴为 off
#         # plt.title('image_fake')  # 图像题目
#         # plt.show()
#         # ------------------------------------------------------
#         return {'A': A, 'B': B, 'B_gray': B_gray, 'label': label,'A_paths': A_path, 'B_paths': B_path}
#
#     def __len__(self):
#         """Return the total number of images in the dataset.
#
#         As we have two datasets with potentially different number of images,
#         we take a maximum of
#         """
#         return max(self.A_size, self.B_size)
