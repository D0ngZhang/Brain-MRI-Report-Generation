
import shutil
import torchio as tio
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# 源文件夹和目标文件夹路径
source_folder = 'E:/data/301'
target_folder = 'E:/data/301/to_be_skullstripped'

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的所有文件
for file in os.listdir(source_folder):
    if file.endswith('T1.nii') or file.endswith('T2.nii'):
        # 构建掩码文件的文件名
        seg_file = file.replace('.nii', '_seg.nii')

        # 确保掩码文件存在
        if os.path.exists(os.path.join(source_folder, seg_file)):
            # 读取图像和掩码文件
            img = tio.ScalarImage(os.path.join(source_folder, file))
            mask = tio.LabelMap(os.path.join(source_folder, seg_file))

            img_temp = img.tensor
            mask = mask.tensor
            try:
                img_temp = img_temp * mask
            except:
                shutil.copy(os.path.join(source_folder, file), os.path.join(target_folder, file))
                shutil.copy(os.path.join(source_folder, seg_file), os.path.join(target_folder, seg_file))
                print(f'Copied {file} and {seg_file} due to shape inconsistency.')
