import csv
import os
import os.path
import sys
import tkinter as tk
from tkinter import messagebox
from tkinter import *
import PIL
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import ttk

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
# import dlib
import matplotlib.pyplot as plt
import cv2
# from imutils.face_utils import rect_to_bb

import Generator__L2
from Generator__L2 import ResidualBlock, Generator_l2, Discriminator
import Generator__G2
from Generator__G2 import ResidualBlock, Generator_g2, Discriminator
sys.path.append("..")

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return x + self.main(x)


class Generator_g2(nn.Module):
    """Generator network."""

    def __init__(self, device, conv_dim=64, c_dim=7, repeat_num=6):

        super(Generator_g2, self).__init__()

        self.device = device
        self.c_dim = c_dim

        # a covariance matrix is represented by value of its eigenvalues and direction of eigenvectors
        self.covariance_angles = nn.Linear(c_dim, 1, bias=False)
        self.covariance_angles.weight.data.fill_(0.0)

        self.covariance_axes = nn.Linear(2, c_dim, bias=False)
        self.covariance_axes.weight.data.fill_(1.0)

        self.mu = nn.Linear(2, c_dim, bias=False)

        layers = []
        layers.append(
            nn.Conv2d(3 + 2, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        )
        layers.append(
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        )
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True)
            )
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(
                nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True)
            )
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        )
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, expr=None, label_trg=None):

        batch_size = x.size(0)

        # case when one watn to reproduce a basic emotion
        if label_trg is not None:

            expr = torch.empty((batch_size, 2), device=self.device)

            for batch_sample in range(batch_size):

                expr[batch_sample, :] = self.mu.weight[label_trg[batch_sample]]

        if expr is None:

            expr = torch.empty((batch_size, 2), device=self.device)
            batch_size = x.size(0)
            if batch_size >= self.c_dim:
                expr[: self.c_dim, :] = torch.tanh(self.mu.weight)
                expr[self.c_dim :] = (
                    torch.rand((batch_size - self.c_dim, 2), device=self.device) * 2 - 1
                )

            else:
                expr[:batch_size, :] = torch.tanh(self.mu.weight[:batch_size])

        # covariance matrix computation C=RDR'

        cos = torch.cos(self.covariance_angles.weight[0])
        sin = torch.sin(self.covariance_angles.weight[0])
        C = torch.empty(7, 2, 2, device=self.device)
        C[:, 0, 0] = (
            torch.abs(self.covariance_axes.weight[:, 0]) * cos * cos
            + torch.abs(self.covariance_axes.weight[:, 1]) * sin * sin
        )
        C[:, 0, 1] = (
            torch.abs(self.covariance_axes.weight[:, 0]) * cos * sin
            - torch.abs(self.covariance_axes.weight[:, 1]) * sin * cos
        )
        C[:, 1, 0] = (
            torch.abs(self.covariance_axes.weight[:, 0]) * sin * cos
            - torch.abs(self.covariance_axes.weight[:, 1]) * cos * sin
        )
        C[:, 1, 1] = (
            torch.abs(self.covariance_axes.weight[:, 0]) * sin * sin
            + torch.abs(self.covariance_axes.weight[:, 1]) * cos * cos
        )
        C_inv = torch.inverse(C)

        # vector of un-normalized distances
        rep_mu = torch.tanh(self.mu.weight.unsqueeze(0).repeat(x.size(0), 1, 1))
        rep_expr = expr.unsqueeze(1).repeat(1, 7, 1)
        vector = rep_expr - rep_mu

        mahalanobis_distances = torch.empty(batch_size, self.c_dim, device=self.device)

        # mahalanobis_distance^2=vec*C_inv*vec
        for ex in range(self.c_dim):
            mahalanobis_distances[:, ex] = (
                vector[:, ex, 0] * vector[:, ex, 0] * C_inv[ex, 0, 0]
                + vector[:, ex, 1] * vector[:, ex, 0] * C_inv[ex, 1, 0]
                + vector[:, ex, 0] * vector[:, ex, 1] * C_inv[ex, 0, 1]
                + vector[:, ex, 1] * vector[:, ex, 1] * C_inv[ex, 1, 1]
            )

        # reshaping expr
        expr2 = expr.view(x.size(0), 2, 1, 1)
        expr3 = expr2.repeat(1, 1, x.size(2), x.size(3))
        # produce imagines
        x = torch.cat([x, expr3], dim=1)
        return self.main(x), mahalanobis_distances, expr

    def print_expr(self):

        print("MU")
        print(torch.tanh(self.mu.weight))
        print("Covariance Matrix angles")
        print(self.covariance_angles.weight)
        print("covariance matrix axes")
        print(self.covariance_axes.weight)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(
            curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(curr_dim, c_dim + 2, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))



class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return x + self.main(x)


class Generator_l2(nn.Module):
    """Generator network."""

    def __init__(self, device, conv_dim=64, c_dim=8, repeat_num=6, n_r=5):
        super(Generator_l2, self).__init__()

        self.nr = n_r
        self.c_dim = c_dim
        self.device = device
        # the six axes, real weight are 6X2
        self.axes = nn.Linear(2, c_dim - 1)
        # make the weight small so that they can easily modified by gradient descend
        self.axes.weight.data = self.axes.weight.data * 0.0001

        layers = []
        layers.append(
            nn.Conv2d(3 + 2, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        )
        layers.append(
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        )
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True)
            )
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(
                nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True)
            )
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        )
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(
        self,
        x,
        c,
        expr_strength,
        mode="train",
        manual_expr=None,
    ):

        """
        mode can be:
            1) random: code is completely random
            2) manual_selection: code is given manually
            3) train: first nr direction ar choosen randomly
            4) test: no direction is choosen randomly
        """

        if mode == "random":

            n_random = x.size(0)
            angle = torch.rand(n_random, device=self.device) * (2 * np.pi)

            expr_strength = torch.rand(n_random, device=self.device)

            random_vector = torch.empty((n_random, 2), device=self.device)

            random_vector[:, 0] = torch.cos(angle) * expr_strength[:n_random]
            random_vector[:, 1] = torch.sin(angle) * expr_strength[:n_random]

            expr2 = random_vector.view(c.size(0), 2, 1, 1)
            expr3 = expr2.repeat(1, 1, x.size(2), x.size(3))

            x = torch.cat([x, expr3], dim=1)
            return self.main(x), random_vector

        else:

            axes_normalized = nn.functional.normalize(self.axes.weight, p=2, dim=1)

            # axis selection
            if not mode == "manual_selection":
                axis = torch.mm(
                    c[:, 1 : self.c_dim], axes_normalized
                )  # axis 0 is neutral and so must be set to 0

            if mode == "train":
                expr = (axis.transpose(0, 1) * expr_strength).transpose(
                    0, 1
                ) + torch.randn(c.size(0), 2, device=self.device) * 0.075
                if x.size(0) >= self.nr:
                    n_random = min(self.nr, x.size(0))
                    angle = torch.rand(n_random, device=self.device) * (2 * np.pi)
                    random_vector = torch.empty((n_random, 2), device=self.device)

                    random_vector[:, 0] = torch.cos(angle) * expr_strength[:n_random]
                    random_vector[:, 1] = torch.sin(angle) * expr_strength[:n_random]

                    expr[:n_random, :] = random_vector

            elif mode == "manual_selection":
                expr = manual_expr

            elif mode == "test":
                expr = (axis.transpose(0, 1) * expr_strength).transpose(0, 1)

            else:

                sys.exit(
                    "Modality can be only 'random','manual_selection','train','test'."
                )

            expr2 = expr.view(x.size(0), 2, 1, 1)  # put c.size(0) if bug!!!!!!!
            expr3 = expr2.repeat(1, 1, x.size(2), x.size(3))

            x = torch.cat([x, expr3], dim=1)
            return self.main(x), expr

    def print_axes(self):

        print("AXES")
        print(nn.functional.normalize(self.axes.weight, p=2, dim=1))


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(
            curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        self.conv3 = nn.Conv2d(curr_dim, 2, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        out_expr_strength = self.conv3(h)
        return (
            out_src,
            out_cls.view(out_cls.size(0), out_cls.size(1)),
            out_expr_strength.view(out_expr_strength.size(0), 2),
        )








translated_img_name = ''
class GANmut:

    def __init__(self, G_path, model='linear', g_conv_dim=64, c_dim=7, g_repeat_num=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", self.device)
        self.model = model


        if self.model == 'linear':
            self.G = Generator_l2(self.device, g_conv_dim, c_dim, g_repeat_num)

        elif self.model == 'gaussian':
            self.G = Generator_g2(self.device, g_conv_dim, c_dim, g_repeat_num)

        else:
            raise ValueError("choose either model='linear' or model='gaussian'")

        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.G.to(self.device)
        self.detector = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        # self.detector2 = dlib.get_frontal_face_detector()
        # self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # self.detector = cv2.CascadeClassifier(cv2.data.haarcascade_frontalface_default.xml)
        # self.detector = dlib.get_frontal_face_detector()


    def emotion_edit(self, img_path, x=None, y=None, theta=None, rho=None, save=False):

        if self.model == 'linear':
            assert (rho is not None) or (theta is not None), 'if model is linear you must provide rho and theta'
        else:
            assert (x is not None) and (y is not None), 'if model is gaussian you must provide x and y'

        # 1 значит, что мы читаем изображение в RGB, то есть цветное
        img = cv2.imread(img_path, 1)  # BGR
        print('emotion_edit path: ', img_path)
        # upload_img_path = 'uploaded_img/uploaded_image.jpg'
        # cv2.imwrite(upload_img_path, img)
        # img_in_project = cv2.imread(upload_img_path, 1)
        #
        # print('EMOTION-edit: uploaded in: ')
        # print(upload_img_path)

        img_rgb = img[:, :, [2, 1, 0]]
        plt.title('Original Image')
        # plt.imshow(img_rgb)


        # extract face
        # faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces = self.detector.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # det = self.detector2(img, 1)[0]
        # print(det)
        # (xx, yy, w, h) = rect_to_bb(det)
        (xx, yy, w, h) = faces[0]
        face = cv2.resize(img[yy:yy + h, xx:xx + w], (128, 128))

        plt.figure()
        plt.title('Detected face')
        plt.imshow(face[:, :, [2, 1, 0]])

        # adapt image format for G
        face = face.transpose((2, 0, 1))  # [H,W,C] --> [C,H,W]
        face = (face / 255.0 - 0.5) / 0.5  # normalize to [-1, 1]
        face = torch.from_numpy(face).float().unsqueeze(0).to(self.device)

        # edit emotion

        with torch.no_grad():

            if self.model == 'linear':
                mode = 'manual_selection'
                expr = (torch.tensor([np.cos(theta), np.sin(theta)]) * rho).to(self.device).float()
                face_g = self.G(face, None, None, mode=mode, manual_expr=expr)[0][0, [2, 1, 0], :, :] / 2 + 0.5
            else:
                expr = torch.Tensor([x, y]).unsqueeze(0).to(self.device)
                face_g = self.G(face, expr)[0][0, [2, 1, 0], :, :] / 2 + 0.5

        face_g = face_g.transpose(0, 2).transpose(0, 1).detach().cpu().numpy()

        plt.figure()
        plt.title('Edited face')
        plt.imshow(face_g)

        # insert edited face in original image
        img_rgb[yy:yy + h, xx:xx + w] = cv2.resize(face_g, (w, h)) * 255

        plt.figure()
        plt.title('Edited image')
        plt.imshow(img_rgb)

        if save:
            save_dir = ""
            if self.model == 'linear':
                # img_name = 'theta_{:0.2f}_rho_{:0.2f}'.format(theta, rho) + os.path.split(img_path)[-1]
                img_name = 'happy' + os.path.split(img_path)[-1]
                save_dir = "\ML2\happy"
                Path(save_dir).mkdir(parents=True, exist_ok=True)
            else:
                # img_name = 'x_{:0.2f}_y_{:0.2f}'.format(x, y) + os.path.split(img_path)[-1]
                img_name = 'sad' + os.path.split(img_path)[-1]
                save_dir = "\ML2\sad"
                Path(save_dir).mkdir(parents=True, exist_ok=True)

            img_name = os.path.join(save_dir, img_name)
            plt.imsave(img_name, img_rgb)
            print(f'edited image saved in {img_name}')
            global translated_img_name
            translated_img_name = img_name


project_path_img = ''
def upload_photo():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        format_img = img.format
        global project_path_img
        project_path_img = f'uploaded_img/for_translation.{format_img}'

        img.save(project_path_img)

        # попробовать сделать запрос размера изображения, возможно в отдельных полях, чтобы пользователь сам вписывал размер изображения.
        # или может быть можно узнать при загрузке размер изображения
        # если размер изображения x=y, то сжимать до квадрата 250 на 250 или 300 на 300
        # если размер изображения х != у, то сжимать до вертикального небольшого размера
        img_resize = img.resize((300, 300))
        photo = ImageTk.PhotoImage(img_resize)
        label_ph = Label(image=photo)
        label_ph.image = photo
        label_ph.place(x=10, y=150)

        # name_img = os.path.split(file_path)[-1]
        # print(name_img)
        # up_img = cv2.imread(name_img, 1)
        # cv2.imshow(up_img)
        # cv2.waitKey(0)
        # cv2.imwrite('for_translation_img.jpg', up_img)




def happy_generation():
    gan_model_happy = GANmut(G_path="models/1800000-G.ckpt", model='gaussian')
    print('happy generation on image for translation in path:')
    print(project_path_img)
    gan_model_happy.emotion_edit(img_path=project_path_img, x=0.31, y=-0.11, save=True)
    translated_sad = Image.open(translated_img_name)
    # попробовать сделать запрос размера изображения, возможно в отдельных полях, чтобы пользователь сам вписывал размер изображения.
    # или может быть можно узнать при загрузке размер изображения
    # если размер изображения x=y, то сжимать до квадрата 250 на 250 или 300 на 300
    # если размер изображения х != у, то сжимать до вертикального небольшого размера
    translated_sad = translated_sad.resize((300, 300))
    translated_sad_ph = ImageTk.PhotoImage(translated_sad)
    label_ph = Label(image=translated_sad_ph)
    label_ph.image = translated_sad_ph
    label_ph.place(x=440, y=150)

def sad_generation():
    gan_model_sad = GANmut(G_path="models/1800000-G.ckpt", model='gaussian')
    print('sad generation on image for translation in path:')
    print(project_path_img)
    gan_model_sad.emotion_edit(img_path=project_path_img, x=-0.8, y=-0.5, save=True)
    translated_sad = Image.open(translated_img_name)
    # попробовать сделать запрос размера изображения, возможно в отдельных полях, чтобы пользователь сам вписывал размер изображения.
    # или может быть можно узнать при загрузке размер изображения
    # если размер изображения x=y, то сжимать до квадрата 250 на 250 или 300 на 300
    # если размер изображения х != у, то сжимать до вертикального небольшого размера
    translated_sad = translated_sad.resize((300, 300))
    translated_sad_ph = ImageTk.PhotoImage(translated_sad)
    label_ph = Label(image=translated_sad_ph)
    label_ph.image = translated_sad_ph
    label_ph.place(x=440, y=150)

def angry_generation():
    gan_model_sad = GANmut(G_path="models/1800000-G.ckpt", model='gaussian')
    print('sad generation on image for translation in path:')
    print(project_path_img)
    gan_model_sad.emotion_edit(img_path=project_path_img, x=0.69, y=-0.76, save=True)
    translated_sad = Image.open(translated_img_name)
    # попробовать сделать запрос размера изображения, возможно в отдельных полях, чтобы пользователь сам вписывал размер изображения.
    # или может быть можно узнать при загрузке размер изображения
    # если размер изображения x=y, то сжимать до квадрата 250 на 250 или 300 на 300
    # если размер изображения х != у, то сжимать до вертикального небольшого размера
    translated_sad = translated_sad.resize((300, 300))
    translated_sad_ph = ImageTk.PhotoImage(translated_sad)
    label_ph = Label(image=translated_sad_ph)
    label_ph.image = translated_sad_ph
    label_ph.place(x=440, y=150)

def disgust_generation():
    gan_model_sad = GANmut(G_path="models/1800000-G.ckpt", model='gaussian')
    print('sad generation on image for translation in path:')
    print(project_path_img)
    gan_model_sad.emotion_edit(img_path=project_path_img, x=-0.75, y=-0.76, save=True)
    translated_sad = Image.open(translated_img_name)
    # попробовать сделать запрос размера изображения, возможно в отдельных полях, чтобы пользователь сам вписывал размер изображения.
    # или может быть можно узнать при загрузке размер изображения
    # если размер изображения x=y, то сжимать до квадрата 250 на 250 или 300 на 300
    # если размер изображения х != у, то сжимать до вертикального небольшого размера
    translated_sad = translated_sad.resize((300, 300))
    translated_sad_ph = ImageTk.PhotoImage(translated_sad)
    label_ph = Label(image=translated_sad_ph)
    label_ph.image = translated_sad_ph
    label_ph.place(x=440, y=150)

def surprise_generation():
    gan_model_sad = GANmut(G_path="models/1800000-G.ckpt", model='gaussian')
    print('sad generation on image for translation in path:')
    print(project_path_img)
    gan_model_sad.emotion_edit(img_path=project_path_img, x=0.75, y=-0.42, save=True)
    translated_sad = Image.open(translated_img_name)
    # попробовать сделать запрос размера изображения, возможно в отдельных полях, чтобы пользователь сам вписывал размер изображения.
    # или может быть можно узнать при загрузке размер изображения
    # если размер изображения x=y, то сжимать до квадрата 250 на 250 или 300 на 300
    # если размер изображения х != у, то сжимать до вертикального небольшого размера
    translated_sad = translated_sad.resize((300, 300))
    translated_sad_ph = ImageTk.PhotoImage(translated_sad)
    label_ph = Label(image=translated_sad_ph)
    label_ph.image = translated_sad_ph
    label_ph.place(x=440, y=150)

def neutral_generation():
    gan_model_sad = GANmut(G_path="models/1800000-G.ckpt", model='gaussian')
    print('sad generation on image for translation in path:')
    print(project_path_img)
    gan_model_sad.emotion_edit(img_path=project_path_img, x=-0.27, y=-0.41, save=True)
    translated_sad = Image.open(translated_img_name)
    # попробовать сделать запрос размера изображения, возможно в отдельных полях, чтобы пользователь сам вписывал размер изображения.
    # или может быть можно узнать при загрузке размер изображения
    # если размер изображения x=y, то сжимать до квадрата 250 на 250 или 300 на 300
    # если размер изображения х != у, то сжимать до вертикального небольшого размера
    translated_sad = translated_sad.resize((300, 300))
    translated_sad_ph = ImageTk.PhotoImage(translated_sad)
    label_ph = Label(image=translated_sad_ph)
    label_ph.image = translated_sad_ph
    label_ph.place(x=440, y=150)

def fear_generation():
    gan_model_sad = GANmut(G_path="models/1800000-G.ckpt", model='gaussian')
    print('sad generation on image for translation in path:')
    print(project_path_img)
    gan_model_sad.emotion_edit(img_path=project_path_img, x=-0.18, y=-0.76, save=True)
    translated_sad = Image.open(translated_img_name)
    # попробовать сделать запрос размера изображения, возможно в отдельных полях, чтобы пользователь сам вписывал размер изображения.
    # или может быть можно узнать при загрузке размер изображения
    # если размер изображения x=y, то сжимать до квадрата 250 на 250 или 300 на 300
    # если размер изображения х != у, то сжимать до вертикального небольшого размера
    translated_sad = translated_sad.resize((300, 300))
    translated_sad_ph = ImageTk.PhotoImage(translated_sad)
    label_ph = Label(image=translated_sad_ph)
    label_ph.image = translated_sad_ph
    label_ph.place(x=440, y=150)

def on_closing():
    if messagebox.askokcancel("Exit", "Do you want to exit?"):
        win.destroy()


global win
win = tk.Tk()
win.protocol("WM_DELETE_WINDOW", on_closing)

win.geometry("750x500")
win.resizable(False, False)
win.title('Generator')

tk.Label(win, text='EMOTION GENERATOR', font=('Times', 14, 'bold')).place(x=270, y=10)
tk.Label(win, text='UPLOAD', font=('Times', 12, 'bold')).place(x=145, y=55)
upload_button = tk.Button(win, text='Upload', bg='#ffb0b0', fg='black', font=('Times', 11, 'bold'), bd=3, width=6, command=upload_photo)
upload_button.place(x=145, y=85)
# блок с выводом загруженной фотографии

# вывод фотографии
# our_image = Image.open("example.jpg")
# our_image = our_image.resize((300,300))
# our_image = ImageTk.PhotoImage(our_image)
# our_label = Label(image=our_image)
# our_label.image = our_image
# our_label.place(x=10, y=150)
# еще один способ вывода
# our_image = Image.open("example.jpg")
# our_image = our_image.resize((300,300))
# our_image = ImageTk.PhotoImage(our_image)
# img_d = canvas.create_image(20, 300, anchor='nw', image=our_image)

tk.Label(win, text='*square or vertical', font=('Times', 9, 'bold')).place(x=130, y=120)

tk.Label(win, text='CHOOSE', font=('Times', 12, 'bold')).place(x=350, y=55)
tk.Label(win, text='EMOTION', font=('Times', 12, 'bold')).place(x=345, y=80)

happy_button = tk.Button(win, text='Happy', bg='#ffb0b0', font=('Times', 11, 'bold'), fg='black', bd=3, width=6, command=happy_generation)
happy_button.place(x=350, y=120)
sad_button = tk.Button(win, text='Sad', bg='#ffb0b0', fg='black', font=('Times', 11, 'bold'), bd=3, width=6, command=sad_generation)
sad_button.place(x=350, y=170)
angry_button = tk.Button(win, text='Angry', bg='#ffb0b0', font=('Times', 11, 'bold'), fg='black', bd=3, width=6, command=angry_generation)
angry_button.place(x=350, y=220)
disgust_button = tk.Button(win, text='Disgust', bg='#ffb0b0', fg='black', font=('Times', 11, 'bold'), bd=3, width=6, command=disgust_generation)
disgust_button.place(x=350, y=270)
surprise_button = tk.Button(win, text='Surprise', bg='#ffb0b0', font=('Times', 11, 'bold'), fg='black', bd=3, width=6, command=surprise_generation)
surprise_button.place(x=350, y=320)
neutral_button = tk.Button(win, text='Neutral', bg='#ffb0b0', fg='black', font=('Times', 11, 'bold'), bd=3, width=6, command=neutral_generation)
neutral_button.place(x=350, y=370)
fear_button = tk.Button(win, text='Fear', bg='#ffb0b0', fg='black', font=('Times', 11, 'bold'), bd=3, width=6, command=fear_generation)
fear_button.place(x=350, y=420)

tk.Label(win, text='RESULT', font=('Times', 12, 'bold')).place(x=565, y=55)
# блок с выводом translated фотографии


win.mainloop()