import os
import sys
import random
import time
print(sys.version)

import numpy as np
import SimpleITK as sitk
from glob import glob
from resample_sitk import resample_sitk_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from torch.utils.data import Dataset
import h5py
from tqdm import tqdm

#memory debugger
#from pympler.tracker import SummaryTracker

path = 'Task07_Pancreas/'
trainpath = path + 'imagesTr/'
testpath = path + 'imagesTs/'
labelpath = path + 'labelsTr/'
valpath = path + 'imagesVal/'
vallabelpath = path + 'labelsVal/'

clip_minbound = -100
clip_maxbound = 200

'''
An images shape is always (x,y,z):
    x = height of the image (direction left arm to right arm)
    y = width of the image (direction back to belly)
    z = depth of the image (direction feet to head)

    If this is confusing: you can imagine the x from the patients point of view being their width
'''


class Pancreas(Dataset):
    '''
    todo:
        cache patches # Do we actually need to chache them, cant we keep them in memory for one batch and throw them away...? (as long as we don't need to reconstruct the image) We don't keep the full images in memory

        up/downsample images voxeldistance, the voxel spacing should be [2.5, 0.8, 0.8] # This page should help to do it in nibabel https://nipy.org/nibabel/coordinate_systems.html#the-affine-as-a-series-of-transformations

    '''
    def __init__(self, type='train'):
        if type == 'train':
            self.imgs = self.load_dir(trainpath)
            self.lbls = self.load_dir(labelpath)
        elif type == 'val':
            self.imgs = self.load_dir(valpath)
            self.lbls = self.load_dir(vallabelpath)
        else:
            self.imgs = self.load_dir(testpath)

        assert len(self.imgs) > 0,f'\n\nMake sure your image path is correct. the train path is currently set to {trainpath} but could not find any training images there.'

    def load_dir(self, path):
        '''
        We do not load the *actual* data yet, that is done in resample_sitk_image in __getitem__
        '''
        return glob(path + "*.nii.gz")

    def __getitem__(self, i):
        def clip(img): # we don't change the std (should we?)
            img = np.maximum(img,clip_minbound)
            img = np.minimum(img,clip_maxbound)
            return img

        sample = resample_sitk_image(self.imgs[i], spacing=[0.8, 0.8, 2.5], interpolator='linear')
        sample = sitk.GetArrayFromImage(sample).transpose(1, 2, 0)
        sample = clip(sample)

        label = resample_sitk_image(self.lbls[i], spacing=[0.8, 0.8, 2.5], interpolator='nearest')
        label = sitk.GetArrayFromImage(label).transpose(1, 2, 0)

        return sample, label

    def __len__(self):
        return len(self.imgs)


class BatchCreator:

    def __init__(self, patch_extractor, dataset, target_size, img_probs):
        self.patch_extractor = patch_extractor
        self.target_size = target_size  # size of the output, can be useful when valid convolutions are used

        self.img_probs = img_probs
        self.dataset = dataset
        self.n = len(dataset)
        self.patch_size = self.patch_extractor.patch_size

    def create_image_batch(self, batch_size):
        '''
        returns a np.array of Patch objects of length batch_size
        '''

        patches = np.zeros(batch_size,dtype=object)

        for i in range(0, batch_size):
            random_index = np.random.choice(np.arange(len(self.dataset)), p=self.img_probs)       # pick random image
            img, lbl = self.dataset[random_index]                            # get image and segmentation map
            patch_img, patch_lbl = self.patch_extractor.get_patch(img,lbl)   # when image size is equal to patch size, this line is useless...
            origin = self.patch_extractor.origin
            patch_size = Coord(self.patch_extractor.patch_size)
            patches[i] = Patch(patch_img,patch_lbl,random_index,origin,patch_size,self.dataset)

        return patches

    def get_image_generator(self, batch_size):
        '''
        Returns a generator that will yield image-batches infinitely in the form of a np.array of Patches objects)
        '''
        while True:
            yield self.create_image_batch(batch_size)


class PatchExtractor:

    def __init__(self, patch_size):
        self.patch_size = patch_size #Coord object
        self.origin = Coord((0,0,0))
        #self.x = 0   #deprecated
        #self.y = 0   #deprecated
        #self.z = 0   #deprecated

    def pad_patch(self, image, label, dims):
        '''
        Called when size in z is smaller than patch size in z. The image and label are padded to the patch size in z.
        '''
        # Only padding in z-direction. + 1 in case of uneven size dif.
        dif = self.patch_size[2] - dims[2]
        pad_size = int(dif/2)
        if dif%2 != 0:
            pad_size += 1

        pad_width = ((0, 0), (0, 0), (pad_size, pad_size))

        # Pad with edge values
        image = np.pad(image, pad_width, mode='edge')
        label = np.pad(label, pad_width, mode='edge')
        dims = Coord(image.shape)

        return image, label, dims

    def get_patch(self, image, label, always_pancreas = False):
        '''
        Get a patch of patch_size from input image, along with corresponding label map.
        at least 30% of the time, this image will contain the pancrea
        This function works with image size >= patch_size, and pick random location of the patch inside the image.

        image: a numpy array representing the input image
        label: a numpy array representing the labels corresponding to input image
        '''

        dims = Coord(image.shape)

        if dims[2] < self.patch_size[2]:
            image, label, dims = self.pad_patch(image, label, dims)

        if (always_pancreas or random.random() < 0.30):
            #select pancreas within bounding box
            pan = np.where(label==1) # return all indices of pancreas voxels
            index = random.choice(range(len(pan[0]))) # choose a random pancreas voxel index

            center = Coord((pan[0][index],pan[1][index],pan[2][index]))
            self.origin = center - self.patch_size + Coord.get_random(self.patch_size)
        else:
            # print(dims, patch_size)
            # pick a random location
            #x = random.randint(0, dims[0] - self.patch_size[0])
            #y = random.randint(0, dims[1] - self.patch_size[1])
            #z = random.randint(0, dims[2] - self.patch_size[2])
            #self.origin = Coord((x,y,z))

            self.origin = Coord.get_random(dims - self.patch_size)

        self.origin.lower_bound(0) #make sure the origin is not outside (negative) of the image
        self.origin.upper_bound(dims - self.patch_size)


        patch = image[  self.origin.x : self.origin.x + self.patch_size[0],
                        self.origin.y : self.origin.y + self.patch_size[1],
                        self.origin.z : self.origin.z + self.patch_size[2]]

        target = label[ self.origin.x : self.origin.x + self.patch_size[0],
                        self.origin.y : self.origin.y + self.patch_size[1],
                        self.origin.z : self.origin.z + self.patch_size[2]]

        target.reshape(self.patch_size[0], self.patch_size[1], self.patch_size[2])

        patch_out = patch  # / 255.  # normalize image intensity to range [0., 1.] # Or should we normalize the entire image
        target_out = target

        return patch_out, target_out


class Patch():
    def __init__(self,pimg,plbl,idx,origin,patch_size,dataset):
        #main data
        self.pimg        = pimg  # patch image
        self.plbl        = plbl  # patch label

        #metadata
        self.idx         = idx     # the index of the image in the dataset
        self.origin      = origin  # the coordinates of the full image where the origin of the patch is
        self.patch_size  = patch_size
        self.dataset     = dataset

    def imshow(self):
        '''
        plots the patch and it's original image and labels (including an indicative red bounding box)
        '''
        halfpatch = self.patch_size//2
        img,lbl = self.dataset[self.idx]  # load the original image

        fig,ax = plt.subplots(2,2)

        #normal image
        ax[0,0].set_title("Image",fontsize=10)
        ax[0,0].imshow(img[:,:,self.origin.z+halfpatch.z])  #[x,y,z]
        ax[0,0].add_patch(patches.Rectangle((self.origin.y,self.origin.x),self.patch_size.x,self.patch_size.y,linewidth=1,edgecolor='r',facecolor='none'))

        #normal label
        ax[0,1].set_title("Label",fontsize=10)
        ax[0,1].imshow(lbl[:, :,self.origin.z+halfpatch.z])
        ax[0,1].add_patch(patches.Rectangle((self.origin.y,self.origin.x),self.patch_size.x,self.patch_size.y,linewidth=1,edgecolor='r',facecolor='none'))

        #patch image
        ax[1,0].set_title("Patch Image",fontsize=10)
        ax[1,0].imshow(self.pimg[:, :, halfpatch.z])

        #patch label
        ax[1,1].set_title("Patch Label")
        ax[1,1].imshow(self.plbl[:, :, halfpatch.z])


        plt.show()

    def __str__(self):
        return f'''[Patch object]
    Index: {self.idx}
    Origin : {self.origin}
    Patch Size : {self.patch_size}
'''


class Coord():
    '''
    You can treat this as if it were a regular tuple,
    you can use the + += - -= // operators on this coord as is intuitive
    you can access the variables by Coord_obj.x or Coord_obj[0]

    Remember:
        An images shape is always (x,y,z):
            x = *height* of the image (direction left arm to right arm)
            y = *width* of the image (direction back to belly)
            z = depth of the image (direction feet to head)
trainpath + "pancreas_001.nii.gz"
        If this is confusing: you can imagine the x from the patients point of view being their width
    '''
    def __init__(self,coord):
        if (not isinstance(coord,tuple) and not isinstance(coord,Coord)):
            coord = (coord,coord,coord) # expand the value so Coord(2) will produce the same as Coord((2,2,2)) this is nice because now we can easily do Coord((x,y,z))/2 instead of Coord((x,y,z))/Coord((2,2,2))

        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]

    def get(self):
        return self.x, self.y, self.z

    def lower_bound(self,bound):
        bound = Coord(bound)
        self.x = max(bound.x,self.x)
        self.y = max(bound.y,self.y)
        self.z = max(bound.z,self.z)

    def upper_bound(self,bound):
        bound = Coord(bound)
        self.x = min(bound.x,self.x)
        self.y = min(bound.y,self.y)
        self.z = min(bound.z,self.z)

    @staticmethod
    def get_random(coord):
        x = random.randint(0, coord[0])
        y = random.randint(0, coord[1])
        z = random.randint(0, coord[2])
        return Coord((x,y,z))

    def __add__(self,other):
        other = Coord(other)
        return Coord((self.x+other.x, self.y+other.y, self.z+other.z))

    def __sub__(self,other):
        other = Coord(other)
        return Coord((self.x-other.x, self.y-other.y, self.z-other.z))

    def __floordiv__(self,other):
        other = Coord(other)
        return Coord((self.x//other.x, self.y//other.y, self.z//other.z))

    def __iadd__(self,other):
        other = Coord(other)
        self.x += other.x
        self.y += other.y
        self.z += other.z

    def __isub__(self,other):
        other = Coord(other)
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z

    def __getitem__(self,i):
        #So you can treat Coord as a regular tuple as well
        return [self.x, self.y, self.z][i]

    def __str__(self):
        return f'({self.x}, {self.y}, {self.z})'


start_time = time.time()

### TYPE ###
type = 'train'
############

if type == "train":
    images = glob(trainpath + "*.nii.gz")
elif type == 'val':
    images = glob(valpath + "*.nii.gz")

sizes = []

for image in tqdm(images):
    sitk_image = resample_sitk_image(image, spacing=[0.8, 0.8, 2.5], interpolator='linear')
    sizes.append(np.prod(np.array(sitk_image.GetSize())))

total_size = np.sum(np.array(sizes).astype(np.int64))
probs = sizes/total_size

print(len(probs))

pancreas = Pancreas(type)
batch_size = 10
patch_size = Coord((128, 128, 64) )
#Create a batch generator given a BatchCreator and PatchExtractor object

patchExtractor = PatchExtractor(patch_size)
batchCreator = BatchCreator(patchExtractor, pancreas, patch_size, probs)
batchGenerator = batchCreator.get_image_generator(batch_size)

filename = type + "a.h5"

if os.path.isfile(filename):
    os.remove(filename)

batches = 800


def make_h5():
    count = 0
    with h5py.File(filename) as file:
        for i in tqdm(range(batches)):
            for patch in next(batchGenerator):
                group = file.create_group(str(count))
                img = group.create_dataset("img", data=np.float16(patch.pimg))
                lbl = group.create_dataset("lbl", data=np.uint8(patch.plbl))
                count += 1


def make_h5_groups():
    count = 0
    count_group = 0

    with h5py.File(filename) as file:
        group = file.create_group("Train")
        for i in tqdm(range(batches)):

            if count_group == 100:
                group = file.create_group("Train" + str(count))
                count_group = 0

            for patch in next(batchGenerator):
                group_2 = group.create_group(str(count))
                img = group_2.create_dataset("img", data=np.float16(patch.pimg))
                lbl = group_2.create_dataset("lbl", data=np.uint8(patch.plbl))
                count += 1
                count_group += 1


make_h5()


print("--- Time: {:.3f} sec ---".format(time.time() - start_time))

