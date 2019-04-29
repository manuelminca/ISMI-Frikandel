import os
import sys
import random
import time
print(sys.version)

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from torch.utils.data import Dataset, DataLoader

path = '../../Task07_Pancreas/Task07_Pancreas/'
trainpath = path + 'imagesTr/'
testpath = path + 'imagesTs/'
labelpath = path + 'labelsTr/'

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
	def __init__(self,train):
		self.train = train
		if self.train: 
			self.imgs = self.load_dir(trainpath)
			self.lbls = self.load_dir(labelpath)
		else:
			self.imgs = self.load_dir(testpath)
			
		assert len(self.imgs) > 0,f'\n\nMake sure your image path is correct. the train path is currently set to {trainpath} but could not find any training images there.'
		
	def load_dir(self,path):
		'''
		We do not load the *actual* data yet, that is done with .get_data() in __getitem__
		'''
		return [nib.load(path+file) for file in os.listdir(path) if file[:8] == 'pancreas' and file[-7:] == '.nii.gz']
		
		
	def __getitem__(self,i):
		def normalize(img): # we don't change the std (should we?)
			u,l = np.max(img),np.min(img)
			img = (img - l)/(u-l)
			return img
			
		#print(self.imgs[i].affine)#I think we have to change the affine to change the voxel spacing https://nipy.org/nibabel/coordinate_systems.html#the-affine-as-a-series-of-transformations
								  # or pixdim from the header...
								  

		sample = self.imgs[i].get_fdata(caching='unchanged') #using get_fdata() instead of get_data() makes it easier to predict the return data type . caching='fill' ensures we cache the image
		
		#sitk doesn't work on nibabel images
		#sample = sitk.resample_sitk_image( sample, spacing=[2.5, 0.8, 0.8],  interpolator=linear)
		sample = normalize(sample)
		
		label = self.lbls[i].get_fdata(caching='unchanged') if self.train else None
		#label = sitk.resample_sitk_image( label, spacing=[2.5, 0.8, 0.8],  interpolator=linear)
	
	
		return sample,label
	
	def __len__(self):
			return len(self.imgs)
				
class BatchCreator:

	def __init__(self, patch_extractor, dataset, target_size):
		self.patch_extractor = patch_extractor
		self.target_size = target_size  # size of the output, can be useful when valid convolutions are used

		self.dataset = dataset
		self.n = len(dataset)
		self.patch_size = self.patch_extractor.patch_size
		
		

	def create_image_batch(self, batch_size):
		'''
		returns a np.array of Patch objects of length batch_size
		'''

		patches = np.zeros(batch_size,dtype=object)
		
		for i in range(0, batch_size):
			random_index = np.random.choice(len(self.dataset))   			# pick random image
			img, lbl = self.dataset[random_index]    						# get image and segmentation map
			patch_img, patch_lbl = self.patch_extractor.get_patch(img,lbl)  # when image size is equal to patch size, this line is useless...
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
		
	def get_patch(self, image, label, always_pancreas = False):
		'''
		Get a patch of patch_size from input image, along with corresponding label map.
		at least 30% of the time, this image will contain the pancrea
		This function works with image size >= patch_size, and pick random location of the patch inside the image.

		image: a numpy array representing the input image
		label: a numpy array representing the labels corresponding to input image
		'''
		
		if (always_pancreas or random.random() < 0.30):
			#select pancreas within bounding box
			pan = np.where(label==1) # return all indices of pancreas voxels
			index = random.choice(range(len(pan[0]))) # choose a random pancrea voxel index
			
			center = Coord((pan[0][index],pan[1][index],pan[2][index]))
			correction = self.patch_size//2
			self.origin = center - correction
		else:   
			# pick a random location
			dims = Coord(image.shape)
			x = random.randint(0, dims[0] - self.patch_size[0])
			y = random.randint(0, dims[1] - self.patch_size[1])
			z = random.randint(0, dims[2] - self.patch_size[2])
			self.origin = Coord((x,y,z))
			
		self.origin.lower_bound(0) #make sure the origin is not outside (negative) of the image
		
		patch = image[  self.origin.x:self.origin.x + self.patch_size[0],
						self.origin.y:self.origin.y + self.patch_size[1],
						self.origin.z:self.origin.z + self.patch_size[2]]
		
		target = label[	self.origin.x:self.origin.x + self.patch_size[0],
						self.origin.y:self.origin.y + self.patch_size[1],
						self.origin.z:self.origin.z + self.patch_size[2]]
						
		target.reshape(self.patch_size[0], self.patch_size[1], self.patch_size[2])

		patch_out = patch  # / 255.  # normalize image intensity to range [0., 1.] # Or should we normalize the entire image
		target_out = target

		return patch_out, target_out

class Patch():
	def __init__(self,pimg,plbl,idx,origin,patch_size,dataset):
		self.pimg        = pimg  # patch image
		self.plbl        = plbl  # patch label
		self.idx         = idx   # the index of the image in the dataset
		self.origin      = origin#the coordinates of the full image where the origin of the patch is
		self.patch_size  = patch_size
		self.dataset     = dataset
		
	
	def imshow(self):
		'''
		plots the patch and it's original image and labels (including an indicative red bounding box)
		'''
		halfpatch = self.patch_size//2
		img,lbl = self.dataset[self.idx] # load the original image
		
		fig,ax = plt.subplots(2,2)
		
		#normal image
		ax[0,0].set_title("Image",fontsize=10)
		ax[0,0].imshow(img[:,:,self.origin.z+halfpatch.z]) #[x,y,z]
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
		self.x = max(bound,self.x)
		self.y = max(bound,self.y)
		self.z = max(bound,self.z)
	
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
		
	
			
#Create the dataset and Load the data (headers)
pancreas = Pancreas(train=True)


#Create a batch generator given a BatchCreator and PatchExtractor object
patch_size = Coord((256, 256, 30))
batch_size = 10
patchExtractor = PatchExtractor(patch_size)
batchCreator = BatchCreator(patchExtractor, pancreas, patch_size)
batchGenerator = batchCreator.get_image_generator(batch_size)

#Get one batch from the generator
batch = next(batchGenerator)

#Get the first patch from the batch
patch = batch[0]

#print the meta data of the patch
print(patch)

#Plot the patch (inlcuding the original image)
#patch.imshow()

#Example of looping through a batch
#for patch in next(batchGenerator):
#	patch.imshow()

#Benchmark speed
#a = time.time()
#for i in range(10):
#	batch = next(batchGenerator)
#print(f'10 batches of batch_size {batch_size} took {time.time()-a} ms')

