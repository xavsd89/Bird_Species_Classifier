
import os
import glob
import pandas as pd

original_images_dir = '..\\birds_dataset\\birds'
label_csv_filename = '..\\birds_dataset\\trainLabels_bird.csv'
cleaned_dir = 'clean_dataset'
image_type = 'jpg'

current_dir = os.getcwd()
orgImageDir = os.path.join(current_dir, original_images_dir)
imageFileList = glob.glob(orgImageDir + '/*.' + image_type)
#%%
# create (filename, class) pair
df = pd.read_csv(os.path.join(current_dir, label_csv_filename))
df.head()
df.shape

#%%
train_cplist = df[df.directory == 'train'][['filename', 'label']].to_numpy()
validate_cplist = df[df.directory == 'validate'][['filename', 'label']].to_numpy()

train_unique_class = df[df.directory == 'train'].label.unique()
validate_unique_class = df[df.directory == 'validate'].label.unique()

print('Number of train classes (labels) = ', len(train_unique_class))
print('Number of validate classes (labels) = ', len(validate_unique_class))

#%%
# create the cleaned image directory
if os.path.exists(os.path.join(current_dir, cleaned_dir)) == True:
    print()
    input('Please delete directory({}) before continue.'.format(cleaned_dir))
    
os.mkdir(os.path.join(current_dir, cleaned_dir))      
print('Creating ' + os.path.join(current_dir, cleaned_dir))
#%%
# create the 'classes' directories
clean_dir_path = os.path.join(current_dir, cleaned_dir)
#%%
os.mkdir(os.path.join(clean_dir_path, 'train'))
os.mkdir(os.path.join(clean_dir_path, 'validate'))

# train directory
for dirName in train_unique_class:
    dn = os.path.join(clean_dir_path, 'train')
    dn = os.path.join(dn, dirName)
    os.mkdir(dn)

# validate directory
for dirName in validate_unique_class:
    dn = os.path.join(clean_dir_path, 'validate')
    dn = os.path.join(dn, dirName)
    os.mkdir(dn)
#%%
# copy image to their respective class directory
import shutil
count = 0

for name, label in train_cplist:
    srcfile = os.path.join(current_dir, original_images_dir)
    srcfile = os.path.join(srcfile, name + '.' + image_type)
    dest = os.path.join(current_dir, cleaned_dir)
    dest = os.path.join(dest, 'train')
    dest = os.path.join (dest, label)
    shutil.copy(srcfile, dest)
    count = count + 1
print('{} files were copied'.format(count))
#%%
count = 0
for name, label in validate_cplist:
    srcfile = os.path.join(current_dir, original_images_dir)
    srcfile = os.path.join(srcfile, name + '.' + image_type)
    dest = os.path.join(current_dir, cleaned_dir)
    dest = os.path.join(dest, 'validate')
    dest = os.path.join (dest, label)
    shutil.copy(srcfile, dest)
    count = count + 1

print('{} files were copied'.format(count))
#%%
# do a unique class count
pd.set_option('display.max_rows', 100)
df.label.value_counts()
