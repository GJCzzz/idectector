from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
# dataroot = './dataset/test/'

dataroot = '../organized_dataset'

# # list of synthesis algorithms
# vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
#         'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal']

# # indicates if corresponding testset has multiple classes
# multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

vals = ['adm', 'dalle2', 'ddpm', 'diff-projectedgan', 'diff-stylegan', 
        'iddpm', 'if', 'ldm', 'midjouney', 'pndm', 'projectedgan', 'sdv1', 'sdv2', 
        'stylegan_official_res', 'vqdiffusion']

multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# model
model_path = 'weights/blur_jpg_prob0.5.pth'
