# experiment description of image segmentation using SAM2

tested model: **s,b+.**
chose model:**s(small)**.*somehow it works far better that b+, the reason behind it is yet to find out*.

## experiment design

During the segementation, random colors are used to signify different entities in one picture.And the original picture(unprocessed),the mask and the processed picture are shown in one single .png file in the outputs folder.

## outcome

The untrained model showed certain ability in segmenting single pictures like No.6(man with a suitcase) and No.7(a hamburger). It relatively succeeded in separating the entity from its environmental background. But when a host of entities are involved in the picture, it failed to work them out as one may expect. For instance, in pic No.3(working area), according to the mask it generated there's no even remote distinctions or borderlines between each entity. Be that as it may, when faced with landscape pictures, some main items were recognized(for example,the boat from pic No.8) and segmented neatly though other items of no apparent significance cannot be separated correctly, such as the mountains and trees in pic NO.8. 
