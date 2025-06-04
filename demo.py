import torch
import numpy as np
import os
from PIL import Image
from models.dynamic_channel import set_uniform_channel_ratio, reset_generator
import models
import time

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# settings
# for attributes to use, modify the load_assets() function
config = 'anycost-ffhq-config-f'
assets_dir = 'assets/demo'
n_style_to_change = 12
device = 'cpu'


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

# Thực thi tác vụ trong thread riêng để không bị chặn giao diện người dùng
class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        t1 = time.time()
        ret = self.fn(*self.args, **self.kwargs)
        t2 = time.time()
        self.signals.result.emit((ret, t2 - t1))

# GUI chính của ứng dụng chỉnh sửa khuôn mặt
class FaceEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        # load assets
        self.load_assets()
        # title
        self.setWindowTitle('Face Editing with Anycost GAN')
        # window size
        # self.setGeometry(50, 50, 1000, 800)  # x, y, w, h
        self.setFixedSize(1000, 800)
        # background color
        # p = self.palette()
        # p.setColor(self.backgroundRole(), Qt.white)
        # self.setPalette(p)

        # plot the original image
        self.original_image = QLabel(self)
        self.set_img_location(self.original_image, 100, 72, 360, 360)
        pixmap = self.np2pixmap(self.org_image_list[0])
        self.original_image.setPixmap(pixmap)
        self.original_image_label = QLabel(self)
        self.original_image_label.setText('original')
        self.set_text_format(self.original_image_label, font_size=10)
        self.original_image_label.move(230, 42)

        # display the edited image
        self.edited_image = QLabel(self)
        self.set_img_location(self.edited_image, 540, 72, 360, 360)
        self.projected_image = self.generate_image()
        self.edited_image.setPixmap(self.projected_image)
        self.edited_image_label = QLabel(self)
        self.edited_image_label.setText('projected')
        self.set_text_format(self.edited_image_label, font_size=10)
        self.edited_image_label.move(670, 42)

        # build the sample list
        drop_list = QComboBox(self)
        drop_list.addItems(self.file_names)
        drop_list.currentIndexChanged.connect(self.select_image)
        drop_list.setGeometry(100, 490, 200, 30)
        drop_list.setCurrentIndex(0)
        drop_list_label = QLabel(self)
        drop_list_label.setText('* select sample:')
        self.set_text_format(drop_list_label, 'left', 10)
        drop_list_label.setGeometry(100, 470, 200, 30)

        self.attr_sliders = dict()
        self.slider_value_labels = dict()  # Lưu label phần trăm tương ứng

        for i_slider, key in enumerate(self.direction_dict.keys()):
            # Dấu tick nhỏ
            tick_label = QLabel(self)
            tick_label.setText('|')
            self.set_text_format(tick_label, 'center', 10)
            tick_label.setGeometry(520 + 175, 470 + i_slider * 40 + 9, 50, 20)

            # Tạo slider
            this_slider = QSlider(Qt.Horizontal, self)
            this_slider.setGeometry(520, 470 + i_slider * 40, 400, 30)
            this_slider.setMinimum(-100)
            this_slider.setMaximum(100)
            this_slider.setValue(0)
            self.attr_sliders[key] = this_slider

            # Tạo label tên thuộc tính bên trái
            attr_label = QLabel(self)
            attr_label.setText(key)
            self.set_text_format(attr_label, 'right', 10)
            attr_label.move(520 - 110, 470 + i_slider * 40 + 2)

            # Tạo label hiển thị giá trị phần trăm bên phải slider
            value_label = QLabel(self)
            value_label.setText("0%")
            self.set_text_format(value_label, 'left', 10)
            value_label.setGeometry(520 + 410, 470 + i_slider * 40, 50, 30)
            self.slider_value_labels[key] = value_label

            # Kết nối sự kiện thay đổi giá trị slider để cập nhật label và gọi hàm update
            def make_slider_callback(k=key):
                def on_value_changed():
                    val = self.attr_sliders[k].value()
                    self.slider_value_labels[k].setText(f"{val}%")
                return on_value_changed

            this_slider.valueChanged.connect(make_slider_callback())

            # Kết nối sự kiện thả tay để cập nhật ảnh
            this_slider.sliderReleased.connect(self.slider_update)

        # build models sliders
        base_h = 560
        channel_label = QLabel(self)
        channel_label.setText('Channel:')
        self.set_text_format(channel_label, 'left', 8)
        channel_label.setGeometry(100, base_h + 5, 100, 30)

        self.channel_slider = QSlider(Qt.Horizontal, self)
        self.channel_slider.setGeometry(180, base_h, 210, 30)
        self.channel_slider.sliderReleased.connect(self.model_update)
        self.channel_slider.setMinimum(0)
        self.channel_slider.setMaximum(3)
        self.channel_slider.setValue(3)
        for i, text in enumerate(['1/4', '1/2', '3/4', '1']):
            channel_label = QLabel(self)
            channel_label.setText(text)
            self.set_text_format(channel_label, 'center', 7)
            channel_label.setGeometry(180 + i * 63 - 50 // 2 + 10, base_h + 20, 50, 20)
        
        self.channel_slider.setStyleSheet("""
            QSlider::handle:horizontal {
                background: #0078d7;
                width: 18px;
                height: 18px;
                border-radius: 9px;
                margin: -6px 0;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #ccc;
                border-radius: 3px;
            }
        """)
        
        resolution_label = QLabel(self)
        resolution_label.setText('Resolution:')
        self.set_text_format(resolution_label, 'left', 8)
        resolution_label.setGeometry(100, base_h + 55, 100, 30)

        self.resolution_slider = QSlider(Qt.Horizontal, self)
        self.resolution_slider.setGeometry(180, base_h + 50, 210, 30)
        self.resolution_slider.sliderReleased.connect(self.model_update)
        self.resolution_slider.setMinimum(0)
        self.resolution_slider.setMaximum(3)
        self.resolution_slider.setValue(3)
        for i, text in enumerate(['128', '256', '512', '1024']):
            resolution_label = QLabel(self)
            resolution_label.setText(text)
            self.set_text_format(resolution_label, 'center', 7)
            resolution_label.setGeometry(180 + i * 63 - 50 // 2 + 10, base_h + 70, 50, 20)
        
        self.resolution_slider.setStyleSheet("""
            QSlider::handle:horizontal {
                background: #0078d7;
                width: 18px;
                height: 18px;
                border-radius: 9px;
                margin: -6px 0;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #ccc;
                border-radius: 3px;
            }
        """)


        # build button slider
        self.reset_button = QPushButton('Reset', self)
        self.reset_button.move(100, 700)
        self.reset_button.clicked.connect(self.reset_clicked)

        # build button slider
        self.finalize_button = QPushButton('Finalize', self)
        self.finalize_button.move(280, 700)
        from functools import partial
        self.finalize_button.clicked.connect(partial(self.slider_update, force_full_g=True))

        # add loading gif
        # create label
        self.loading_label = QLabel(self)
        self.loading_label.setGeometry(500 - 25, 240, 50, 50)

        self.loading_label.setObjectName("label")
        self.movie = QMovie(os.path.join(assets_dir, "loading.gif"))
        self.loading_label.setMovie(self.movie)
        self.movie.start()
        self.movie.setScaledSize(QSize(50, 50))
        self.loading_label.setVisible(False)

        # extra time stat
        self.time_label = QLabel(self)
        self.time_label.setText('')
        self.set_text_format(self.time_label, 'center', 10)
        self.time_label.setGeometry(500 - 25, 240, 50, 50)

        # status bar
        self.statusBar().showMessage('Ready.')

        # multi-thread
        self.thread_pool = QThreadPool()

        self.show()

    def load_assets(self):
        self.anycost_channel = 1.0
        self.anycost_resolution = 1024

        # build the generator
        self.generator = models.get_pretrained('generator', config).to(device)
        self.generator.eval()
        self.mean_latent = self.generator.mean_style(10000)

        # select only a subset of the directions to use
        '''
        possible keys:
        ['00_5_o_Clock_Shadow', '01_Arched_Eyebrows', '02_Attractive', '03_Bags_Under_Eyes', '04_Bald', '05_Bangs',
         '06_Big_Lips', '07_Big_Nose', '08_Black_Hair', '09_Blond_Hair', '10_Blurry', '11_Brown_Hair', '12_Bushy_Eyebrows',
         '13_Chubby', '14_Double_Chin', '15_Eyeglasses', '16_Goatee', '17_Gray_Hair', '18_Heavy_Makeup', '19_High_Cheekbones',
         '20_Male', '21_Mouth_Slightly_Open', '22_Mustache', '23_Narrow_Eyes', '24_No_Beard', '25_Oval_Face', '26_Pale_Skin',
         '27_Pointy_Nose', '28_Receding_Hairline', '29_Rosy_Cheeks', '30_Sideburns', '31_Smiling', '32_Straight_Hair',
         '33_Wavy_Hair', '34_Wearing_Earrings', '35_Wearing_Hat', '36_Wearing_Lipstick', '37_Wearing_Necklace',
         '38_Wearing_Necktie', '39_Young']
        '''

        direction_map = {
            'smiling': '31_Smiling',
            'young': '39_Young',
            'wavy hair': '33_Wavy_Hair',
            'gray hair': '17_Gray_Hair',
            'blonde hair': '09_Blond_Hair',
            'eyeglass': '15_Eyeglasses',
            'mustache': '22_Mustache',
        }

        boundaries = models.get_pretrained('boundary', config)
        self.direction_dict = dict()
        for k, v in direction_map.items():
            self.direction_dict[k] = boundaries[v].view(1, 1, -1)

        # 3. prepare the latent code and original images
        file_names = sorted(os.listdir(os.path.join(assets_dir, 'input_images')))
        self.file_names = [f for f in file_names if f.endswith('.png') or f.endswith('.jpg')]
        self.latent_code_list = []
        self.org_image_list = []

        for fname in self.file_names:
            org_image = np.asarray(Image.open(os.path.join(assets_dir, 'input_images', fname)).convert('RGB'))
            latent_code = torch.from_numpy(
                np.load(os.path.join(assets_dir, 'projected_latents',
                                     fname.replace('.jpg', '.npy').replace('.png', '.npy'))))
            self.org_image_list.append(org_image)
            self.latent_code_list.append(latent_code.view(1, -1, 512))

        # set up the initial display
        self.sample_idx = 0
        self.org_latent_code = self.latent_code_list[self.sample_idx]

        # input kwargs for the generator
        self.input_kwargs = {'styles': self.org_latent_code, 'noise': None, 'randomize_noise': False,
                             'input_is_style': True}

    @staticmethod
    def np2pixmap(np_arr):
        height, width, channel = np_arr.shape
        q_image = QImage(np_arr.data, width, height, 3 * width, QImage.Format_RGB888)
        return QPixmap(q_image)

    @staticmethod
    def set_img_location(img_op, x, y, w, h):
        img_op.setScaledContents(True)
        img_op.setFixedSize(w, h)  # w, h
        img_op.move(x, y)  # x, y

    @staticmethod
    def set_text_format(text_op, align='center', font_size=15):
        if align == 'center':
            align = Qt.AlignCenter
        elif align == 'left':
            align = Qt.AlignLeft
        elif align == 'right':
            align = Qt.AlignRight
        else:
            raise NotImplementedError
        text_op.setAlignment(align)
        text_op.setFont(QFont('Arial', font_size))

    def select_image(self, idx):
        self.sample_idx = idx
        self.org_latent_code = self.latent_code_list[self.sample_idx]
        pixmap = self.np2pixmap(self.org_image_list[self.sample_idx])
        self.original_image.setPixmap(pixmap)
        self.input_kwargs['styles'] = self.org_latent_code
        self.projected_image = self.generate_image()
        self.edited_image.setPixmap(self.projected_image)
        self.reset_sliders()

    def reset_sliders(self):
        for slider in self.attr_sliders.values():
            slider.setValue(0)
        self.edited_image_label.setText('projected')
        self.statusBar().showMessage('Ready.')
        self.time_label.setText('')

    def generate_image(self):
        def image_to_np(x):
            assert x.shape[0] == 1
            x = x.squeeze(0).permute(1, 2, 0)
            x = (x + 1) * 0.5  # 0-1
            x = (x * 255).cpu().numpy().astype('uint8')
            return x

        with torch.no_grad():
            out = self.generator(**self.input_kwargs)[0].clamp(-1, 1)
            out = image_to_np(out)
            out = np.ascontiguousarray(out)
            return self.np2pixmap(out)

    def set_sliders_status(self, active):
        for slider in self.attr_sliders.values():
            slider.setEnabled(active)

    def slider_update(self, force_full_g=False):
        self.set_sliders_status(False)
        self.statusBar().showMessage('Running...')
        self.time_label.setText('')
        self.loading_label.setVisible(True)
        max_value = 0.6
        edited_code = self.org_latent_code.clone()
        for direction_name in self.attr_sliders.keys():
            edited_code[:, :n_style_to_change] = \
                edited_code[:, :n_style_to_change] \
                + self.attr_sliders[direction_name].value() * self.direction_dict[direction_name] / 100 * max_value
        
        def debug_attribute(attr_name, max_val, edited_code):
            val = self.attr_sliders[attr_name].value()
            direction = self.direction_dict[attr_name]
            org_latent = self.org_latent_code

            print(f"========== DEBUG: THANH TRƯỢT {attr_name.upper()} ==========")
            print(f"Giá trị thanh trượt {attr_name}: {val}")
            print(f"Giá trị tối đa chỉnh sửa: {max_val}")
            print(f"Kích thước hướng chỉnh sửa: {direction.shape}")
            print(f"Hướng chỉnh sửa (5 chiều đầu tiên): {direction[0, 0, :5]}")
            print(f"Latent code gốc (5 chiều đầu tiên): {org_latent[0, 0, :5]}")
            print(f"Latent code đã chỉnh sửa (5 chiều đầu tiên): {edited_code[0, 0, :5]}")
            delta = edited_code - org_latent
            print(f"Hiệu chỉnh latent (5 chiều đầu tiên): {delta[0, 0, :5]}")
            print("===========================================\n")

        # Gọi debug cho 2 thuộc tính
        debug_attribute('smiling', 0.6, edited_code)
        #debug_attribute('young', 0.6, edited_code)
        #debug_attribute('wavy hair', 0.6, edited_code)
        #debug_attribute('gray hair', 0.6, edited_code)
        #debug_attribute('blonde hair', 0.6, edited_code)
        #debug_attribute('eyeglass', 0.6, edited_code)
        #debug_attribute('mustache', 0.6, edited_code)
        self.input_kwargs['styles'] = edited_code
        if not force_full_g:
            set_uniform_channel_ratio(self.generator, self.anycost_channel)
            self.generator.target_res = self.anycost_resolution
        # generate the images in a separate thread
        worker = Worker(self.generate_image)
        worker.signals.result.connect(self.after_slider_update)
        self.thread_pool.start(worker)
        

    def after_slider_update(self, ret):
        edited, used_time = ret
        self.edited_image.setPixmap(edited)

        reset_generator(self.generator)
        self.edited_image_label.setText('edited')
        self.statusBar().showMessage('Done in {:.2f}s'.format(used_time))
        self.time_label.setText('{:.2f}s'.format(used_time))
        self.set_sliders_status(True)
        self.loading_label.setVisible(False)

    def model_update(self):
        self.anycost_channel = [0.25, 0.5, 0.75, 1.0][self.channel_slider.value()]
        self.anycost_resolution = [128, 256, 512, 1024][self.resolution_slider.value()]

    def reset_clicked(self):
        self.reset_sliders()
        self.edited_image.setPixmap(self.projected_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceEditor()
    sys.exit(app.exec_())
