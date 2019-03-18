import os
import cv2

#annotation
#image_name face_id_in_image face_box_top face_box_left face_box_right face_box_bottom face_box_cofidence expression_label

expw_image_basepath = 'E:/work_dir/expression_database/ExpW/image/origin/'
expw_label_file = 'E:/work_dir/expression_database/ExpW/label/label.lst'
expw_dstimage_basepath = 'E:/work_dir/expression_database/expression_face/ExpW/'

expression_ind_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def prepareLabelFile():
    fp = open(expw_label_file, 'r')
    data = fp.readlines()
    line_length = len(data)
    for data_temp in data:
        data_split = data_temp.split(' ')
        file_name = data_split[0]
        btop = int(data_split[2])
        bleft = int(data_split[3])
        bright = int(data_split[4])
        bbottom = int(data_split[5])
        exp_label = int(data_split[7])
        file_path = expw_image_basepath+file_name
        dst_file_path = expw_dstimage_basepath+expression_ind_dict[exp_label]+'/'+file_name

        image = cv2.imread(file_path)
        cv2.imwrite(dst_file_path, image)
        face_roi = image[btop:bbottom, bleft:bright]
        cv2.imwrite(dst_file_path, face_roi)

    fp.close()


if __name__ == '__main__':
    prepareLabelFile()

