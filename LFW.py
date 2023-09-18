import cv2
import os

# make_dataset_name_list(path) ==============================
# path (string) : directory of the dataset
# return (list) - list of name without extension
# ===========================================================
def make_dataset_name_list(path):
    name_list = os.listdir(path_dir)
    print(name_list[0])  # 'Aaron_Eckhart_0001.jpg'
    print(len(name_list))  # 9330

    name_list_nonex = []
    for i in range(len(name_list)):
        name_list_nonex.append(name_list[i].replace(".jpg", ""))
    print(name_list_nonex[0])  # 'Aaron_Eckhart_0001'
    
    return name_list_nonex

# find_face(image, model) ===================================
# image (ndarray) : image array
# model (string) : directory of the xml file
# return (ndarray) - [[x, y, w, h], ...]
# ===========================================================
def find_face(image, model):
    face_cascade = cv2.CascadeClassifier(model)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(image_gray, 1.3, 5) # image, scaleFactor : size of rect(smaller), minNeighbors : minimum number of adjacent rects that accept as a face

# save_rect(image, rects, show) =============================
# image (ndarray) : image array
# rects (ndarray) : captured rect arrays [[x, y, w, h], ...]
# show (bool) : if show output image (=False)
# return (None)
# ===========================================================
def save_rect(image, rects, show=False):
    for i in range(len(rects)):
        x, y, w, h = rects[i]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite(f"outputs/LFW/{name}_{i}.jpg", image)
        
        if show:
            cv2.imshow("face_recongnition", image)
            cv2.waitKey(0)  # waitKey(time) : wait until a specific key pressed down for time ms
            cv2.destroyAllWindows()


path_dir = "datasets/LFW/"
haarcascade = "models/haarcascade_frontalface_default.xml"

image_name_list = make_dataset_name_list(path_dir)

for name in image_name_list[:3]:
    face = cv2.imread("datasets/LFW/" + name + ".jpg")
    face_rects = find_face(face, haarcascade)  #[[x y w h]]
    save_rect(face, face_rects, show=True)