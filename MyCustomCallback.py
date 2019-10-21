from myMetrics import *
from Utils import *
from tqdm import tqdm
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend
from sklearn.metrics import confusion_matrix


# class MyCustomCallback(tf.keras.callbacks.Callback):
class CallbackmIoU(Callback):

    def __init__(self, path, lr):
        super().__init__()
        self.mIoU = []
        self.epoch = []
        self.path = path
        self.init_lr = lr

        self.mapLabel = [
            (0, 0, 0),
            (0, 0, 128),
            (0, 128, 0),
            (0, 128, 128),
            (128, 0, 0),
            (128, 0, 128),
            (128, 128, 0),
            (128, 128, 128),
            (0, 0, 64),
            (0, 0, 192),
            (0, 128, 64),
            (0, 128, 192),
            (128, 0, 64),
            (128, 0, 192),
            (128, 128, 64),
            (128, 128, 192),
            (0, 64, 0),
            (0, 64, 128),
            (0, 192, 0),
            (0, 192, 128),
            (128, 64, 0),
        ]

        self.img_dict = {
            "2008_000045.png": "Treno",
            "2008_000093.png": "Divano",
            "2008_000142.png": "Persona e cavallo",
            "2008_000689.png": "Moto",
            "2008_000585.png": "Aereo",
            "2008_001047.png": "Barca",
            "2008_001704.png": "Schermo",
            "2008_001770.png": "Uccello",
            "2008_002062.png": "Macchina",
            "2008_002583.png": "Gatto"
        }

    # def on_train_batch_begin(self, batch, logs=None):
    #     print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    def on_train_batch_begin(self, batch, logs=None):

        current_iter = tf.keras.backend.eval(self.model.optimizer.iterations)

        if current_iter != 0:

            step = self.params['steps']
            epoch_sz = self.params['epochs']
            max_iter = epoch_sz * step
            # max_iter = 50000
            up = (1 - (current_iter / max_iter)) ** 0.9
            new = self.init_lr * up

            tf.keras.backend.set_value(self.model.optimizer.lr, new)

            if current_iter % step == 0:
                print('\nIteration %05d: reducing learning '
                      'rate to %s.' % (current_iter, new))

        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.init_lr)
            print('\nIteration %05d: reducing learning '
                  'rate to %s.' % (current_iter, self.init_lr))

    def on_epoch_begin(self, epoch, logs=None):
        # def on_train_batch_end(self, batch, logs=None):
        # if ( epoch % 2 == 0 and epoch != 0 )or epoch == self.params['epochs']:
        if epoch % 2 == 0 or epoch == self.params['epochs']:
            # if True:

            print("Begin-Callback")

            # dir_img = 'Y:/tesisti/rossi/datasetPascal/images_prepped_test/'
            dir_img = 'Y:/tesisti/rossi/data/train_val_test_png/val_png/'
            images = glob.glob(dir_img + "*.png")
            images.sort()

            # dir_seg = 'Y:/tesisti/rossi/datasetPascal/gray/val/'
            dir_seg = 'Y:/tesisti/rossi/data/segmentation_gray/gray/val/'
            segs = glob.glob(dir_seg + "*.png")
            segs.sort()

            # mat = tf.keras.backend.zeros(shape=(21, 21), dtype="int32")
            mat = np.zeros(shape=(21, 21), dtype=np.int32)

            # for k in range(len(images)):
            for k in tqdm(range(len(images))):
                img = cv2.imread(images[k])
                seg = cv2.imread(segs[k], cv2.IMREAD_GRAYSCALE)

                scale = img / 127.5 - 1
                res = self.model.predict(np.expand_dims(scale, 0))

                labels = np.argmax(res.squeeze(), -1)
                labels = labels.astype(np.uint8)

                if (images[k])[-15:] in self.img_dict:

                    h_z, w_z = labels.shape
                    imgNew = np.zeros((h_z, w_z, 3), np.uint8)
                    for i in range(0, 21):
                        mask = cv2.inRange(labels, i, i)
                        v = self.mapLabel[i]
                        imgNew[mask > 0] = v

                    image = make_image(imgNew)
                    summary = tf.Summary(
                        value=[tf.Summary.Value(tag=self.img_dict.get((images[k])[-15:]), image=image)])
                    writer = tf.summary.FileWriter(self.path + 'logs')
                    writer.add_summary(summary, epoch)
                    writer.close()

                tmp = confusion_matrix(seg.flatten(), labels.flatten(),
                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
                mat = mat + tmp

            iou = compute_and_print_IoU_per_class(confusion_matrix=mat, num_classes=21)

            self.mIoU.append(iou)
            self.epoch.append(epoch)
            print('The mIoU for epoch {} is {:7.2f}.'.format(epoch, iou))

    #
    # def on_test_batch_begin(self, batch, logs=None):
    #     print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
    #
    # def on_test_batch_end(self, batch, logs=None):
    #     print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

    # def on_epoch_end(self, epoch, logs=None):
    #     print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch,
    #     logs['loss'],
    #                                                                                                 logs['mae']))

# class CallbackLr(init_lr=0, tr_sz=4498, bt_sz=12):

# def sched(self, epoch, logs={}):
#     print('epoch {:7.5f} '.format(backend.eval(self.model.optimizer.lr)))
#     # print('iteration : {}'.format(backend.eval(self.model.optimizer.iterations)))
#     # print('learning rate on batch {} is {:7.5f}.'.format(batch, lr))
# class CallbackLr(Callback):
#
#     """Learning rate scheduler.
#         verbose: int. 0: quiet, 1: update messages.
#     """
#
#     def on_epoch_begin(self, epoch, logs=None):
#
#         bt_sz = self.params['batch_size']
#         print('bt_size '+str(bt_sz))
#
#         # lr = float(backend.get_value(self.model.optimizer.lr))


#     backend.set_value(self.model.optimizer.lr, lr)
#     if self.verbose > 0:
#         print('\nEpoch %05d: LearningRateScheduler reducing learning '
#               'rate to %s.' % (epoch + 1, lr))
#
# def on_epoch_end(self, epoch, logs=None):
#     logs = logs or {}
#     logs['lr'] = backend.get_value(self.model.optimizer.lr)
