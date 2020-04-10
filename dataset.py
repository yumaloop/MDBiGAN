import cv2
import glob
import numpy as np
import skvideo.io
import torch
import torchvision

class WeizmannHumanActionImage(torch.utils.data.Dataset):
    def __init__(self, 
                 path="weizmann-human-action", 
                 trans_label=None,
                 trans_data=torchvision.transforms.ToTensor(), 
                 train=True):
        
        self.trans_label = trans_label
        self.trans_data = trans_data
        self.train = train
        
        labelset = []
        videoset = []
        dataset = []

        v_id = 0
        for class_label, class_dir in enumerate(glob.glob(path+"/*")):
            for file_path in glob.glob(class_dir+"/*.avi"):
                video = skvideo.io.vread(file_path)
                for img in video:
                    img = cv2.resize(img , (96, 96))
                    labelset.append(class_label)
                    videoset.append(v_id)
                    dataset.append(img)
                v_id += 1
                    
        self.labelset = np.array(labelset)
        self.videoset = np.array(videoset)
        self.dataset = np.array(dataset)
        self.datanum = len(dataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_label = self.labelset[idx]
        out_id    = self.videoset[idx]
        out_data  = self.dataset[idx]

        if self.trans_data:
            out_data = self.trans_data(out_data)


        return out_data, out_id, out_label


class WeizmannHumanActionVideo(torch.utils.data.Dataset):
    def __init__(self, 
                 path="weizmann-human-action", 
                 trans_label=None,
                 trans_data=torchvision.transforms.ToTensor(), 
                 frame_size=(96, 96), train=True):
        
        self.trans_label = trans_label
        self.trans_data = trans_data
        self.train = train
        
        labelset = []
        dataset = []

        for class_label, class_dir in enumerate(glob.glob(path+"/*")):
            for filepath in glob.glob(class_dir+"/*.avi"):
                video = skvideo.io.vread(filepath)
                video = self.preprocess_video(video, frame_size)
                labelset.append(class_label)
                dataset.append(video)

        self.labelset = np.array(labelset)
        self.dataset = np.array(dataset)
        self.datanum = len(dataset)
        
    def preprocess_video(self, video, frame_size=(96, 96)):
        video_resize=[]
        for img in video:
            img = cv2.resize(img, frame_size)
            img = img.transpose(2, 1, 0) # (channel, height, width)
            img = img / 255.
            video_resize.append(img)
        video_resize = np.array(video_resize).astype(np.float32)
        return video_resize

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_label = self.labelset[idx]
        out_data  = self.dataset[idx]
        if self.trans_data:
            out_data = self.trans_data(out_data)
        return out_data, out_label
