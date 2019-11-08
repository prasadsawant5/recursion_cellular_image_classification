import os
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet import nd
from mxnet.io import ImageRecordIter
from PIL import Image
import pydicom
import cv2
from tqdm import tqdm

class MxNetTrainer:
    def __init__(self, hemorrhage_type):
        self.hemorrhage_type = hemorrhage_type

        if os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rec'))):
            self.rec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rec'))
        else:
            self.rec_path = None

        self.save_path = None

        if not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'mxnet', hemorrhage_type))):
            os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'mxnet', hemorrhage_type)))

        self.save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'mxnet', hemorrhage_type))

        self.test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stage_1_test_images'))

        self.net = gluon.nn.Sequential()        

        self.ctx = mx.gpu()
        self.num_outputs = 2


        self.epochs = 6
        self.learning_rate = 1e-3
        self.batch_size = 32

        if self.rec_path != None:
            if self.hemorrhage_type == 'epidural':
                self.train_data = ImageRecordIter(
                    path_imgrec = os.path.join(self.rec_path, 'epidural_rec.rec'),
                    path_imgidx = os.path.join(self.rec_path, 'epidural_rec.idx'),
                    data_shape = (3, 384, 384),
                    batch_size = self.batch_size,
                    shuffle = True
                )
            elif self.hemorrhage_type == 'intraparenchymal':
                self.train_data = ImageRecordIter(
                    path_imgrec = os.path.join(self.rec_path, 'intraparenchymal_rec.rec'),
                    path_imgidx = os.path.join(self.rec_path, 'intraparenchymal_rec.idx'),
                    data_shape = (3, 384, 384),
                    batch_size = self.batch_size,
                    shuffle = True
                )
            elif self.hemorrhage_type == 'intraventricular':
                self.train_data = ImageRecordIter(
                    path_imgrec = os.path.join(self.rec_path, 'intraventricular_rec.rec'),
                    path_imgidx = os.path.join(self.rec_path, 'intraventricular_rec.idx'),
                    data_shape = (3, 384, 384),
                    batch_size = self.batch_size,
                    shuffle = True
                )
            elif self.hemorrhage_type == 'subarachnoid':
                self.train_data = ImageRecordIter(
                    path_imgrec = os.path.join(self.rec_path, 'subarachnoid_rec.rec'),
                    path_imgidx = os.path.join(self.rec_path, 'subarachnoid_rec.idx'),
                    data_shape = (3, 384, 384),
                    batch_size = self.batch_size,
                    shuffle = True
                )
            elif self.hemorrhage_type == 'subdural':
                self.train_data = ImageRecordIter(
                    path_imgrec = os.path.join(self.rec_path, 'subdural_rec.rec'),
                    path_imgidx = os.path.join(self.rec_path, 'subdural_rec.idx'),
                    data_shape = (3, 384, 384),
                    batch_size = self.batch_size,
                    shuffle = True
                )

    def model(self):
        with self.net.name_scope():
            self.net.add(gluon.nn.Conv2D(channels=32, kernel_size=5, activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            self.net.add(gluon.nn.Conv2D(channels=64, kernel_size=5, activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            self.net.add(gluon.nn.Conv2D(channels=128, kernel_size=5, activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            self.net.add(gluon.nn.Flatten())

            self.net.add(gluon.nn.Dense(256, 'relu'))
            self.net.add(gluon.nn.Dense(64, 'relu'))
            self.net.add(gluon.nn.Dense(32, 'relu'))
            
            self.net.add(gluon.nn.Dense(self.num_outputs))

    def evaluate_accuracy(self, data, label):
        acc = mx.metric.Accuracy()
        output = self.net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)

        return acc.get()[1]

    def train(self):
        self.model()

        self.net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx)
        
        smoothing_constant = .01

        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()    

        trainer = gluon.Trainer(self.net.collect_params(), 'adam', {'learning_rate': self.learning_rate})

        for e in range(self.epochs):
            i = 0
            self.train_data.reset()
            while self.train_data.iter_next():
                d = self.train_data.getdata() / 255.
                l = self.train_data.getlabel()

                data = d.as_in_context(self.ctx)
                label = l.as_in_context(self.ctx)

                step = data.shape[0]
                with autograd.record():
                    output = self.net(data)
                    loss = softmax_cross_entropy(output, label)
                
                loss.backward()

                trainer.step(step)
                ##########################
                #  Keep a moving average of the losses
                ##########################
                curr_loss = nd.mean(loss).asscalar()
                moving_loss = (curr_loss if ((i == 0) and (e == 0))
                    else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
                

                acc = self.evaluate_accuracy(data, label)
                print("Epoch {:03d} ... Dataset {:03d} ... ".format(e+1, i), "Loss = {:.4f}".format(curr_loss), " Moving Loss = {:.4f}".format(moving_loss), " Accuracy = {:.4f}".format(acc))

                # self.summary_writer.add_histogram(tag='accuracy', values=acc, global_step=e)

                i += 1

            # self.summary_writer.add_scalar(tag='moving_loss', value=moving_loss, global_step=e)

        self.save_path = os.path.join(self.save_path, 'model.params')
        self.net.save_parameters(self.save_path)


    def inference(self):
        model_path = os.path.join(self.save_path, 'model.params')

        if os.path.exists(model_path):
            # if not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test'))):
            #     os.mkdir(os.path.join(os.path.dirname(__file__), '..', 'test'))

            # for i in tqdm(os.listdir(self.test_dir)):
            #     img = pydicom.dcmread(os.path.join(self.test_dir, i)).pixel_array

            #     f_name = i.split('.')[0]

            #     cv2.imwrite(os.path.join(os.path.dirname(__file__), '..', 'test', f_name + '.jpg'), img)

            # img = pydicom.dcmread(os.path.join(self.test_dir, os.listdir(self.test_dir)[0])).pixel_array

            img = cv2.imread(os.path.join(os.path.dirname(__file__), '..', 'test', os.listdir(os.path.join(os.path.dirname(__file__), '..', 'test'))[0]))
            img_list = []

            self.model()

            self.net.load_parameters(model_path, ctx=self.ctx)

            img = nd.array(cv2.resize(img, (384,384)) / 255.)

            img_list.append(img)
            img_list = np.array(img_list)

            data = img_list.as_in_context(self.ctx)

            with autograd.record():
                output = self.net(data)

                print(output)
    