# ISMI-Frikandel

todo:
 - cache patches. (Do we actually need to cache *patches*? cant we keep them in memory for one batch and throw them away...? (as long as we don't need to reconstruct the image) We don't keep the full images in memory)
	 	
 - up/downsample images voxeldistance. the voxel spacing should be [2.5, 0.8, 0.8] 
This page should help to do it in nibabel https://nipy.org/nibabel/coordinate_systems.html#the-affine-as-a-series-of-transformations

issues:
 - High memory usage
