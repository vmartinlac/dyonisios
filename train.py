import cv2
import time
import math
import re
import random
import numpy
import sqlite3
import sys
import os
import os.path
import torch
import torchvision.transforms

class Database:

    def __init__(self, root_directory):
        self.root_directory = sys.argv[1]
        db_filename = os.path.join(self.root_directory, "db.sqlite")
        self.db = sqlite3.connect(db_filename)

        cur0 = self.db.cursor()
        cur0.execute("SELECT id FROM classes")
        self.class_ids = [ int(row[0]) for row in cur0 ]
        cur0.close()
        del cur0

        self.class_id_to_index = dict()
        i = 0
        for id_ in self.class_ids:
            self.class_id_to_index[id_] = i
            i += 1

    def create_dataset(self, name, samples_per_class):

        cur0 = self.db.cursor()
        cur0.execute("SELECT id FROM classes")

        dataset = dict()

        for row in cur0:
            class_id = row[0]
            cur1 = self.db.cursor()
            cur1.execute( "SELECT id FROM samples WHERE class_id=? AND id NOT IN ( SELECT sample_id FROM memberships )", (class_id,) )
            ids = numpy.asarray( cur1.fetchall(), dtype=numpy.int32 )
            if ids.shape[0] < samples_per_class:
                raise Exception("Not enough samples available!")
            numpy.random.shuffle(ids)
            dataset[class_id] = ids[:samples_per_class, 0]
            cur1.close()
            del cur1

        cur0.execute( "INSERT INTO datasets(name) VALUES(?)", (name,) )
        dataset_id = cur0.lastrowid
        for class_id in dataset:
            cur1 = self.db.cursor()
            cur1.executemany( "INSERT INTO memberships(sample_id, dataset_id) VALUES(?,?)", [ (int(x), dataset_id) for x in dataset[class_id] ] )
            cur1.close()
            del cur1

        self.db.commit()

    def __del__(self):
        self.db.close()

class Dataset(torch.utils.data.Dataset):

    def __init__(self, database, dataset_name):
        self.database = database
        self.dataset_name = dataset_name

        cur = self.database.db.cursor()
        cur.execute("SELECT filename, class_id FROM samples WHERE id in (SELECT sample_id FROM memberships WHERE dataset_id IN (SELECT id FROM datasets WHERE name=?))", (self.dataset_name,))
        self.table = cur.fetchall()
        cur.close()
        del cur

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):

        item = self.table[i]

        image_filename = os.path.join(self.database.root_directory, item[0])

        # load image.
        im = cv2.imread(image_filename)

        # convert to RGB.
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # convert to floating point.
        im = numpy.asarray(im, dtype=numpy.float32)

        # normalize.
        im /= 255.0

        mean = [0.485, 0.456, 0.406]
        sigma = [0.229, 0.224, 0.225]

        im[:,:,0] = (im[:,:,0] - mean[0]) / sigma[0]
        im[:,:,1] = (im[:,:,1] - mean[1]) / sigma[1]
        im[:,:,2] = (im[:,:,2] - mean[2]) / sigma[2]

        # make pytorch tensor.

        im = torch.Tensor(im)
        im = im.permute([2, 0, 1]);

        index = self.database.class_id_to_index[item[1]]

        return (im, index)

class Trainer:

    def __init__(self, database):

        self.database = database

        self.learning_rate = 0.0001
        #self.learning_rate = 0.00003
        self.train_batch_size = 64
        self.test_batch_size = 64

        self.device = 'cuda'

        self.train_loader = torch.utils.data.DataLoader( Dataset(self.database, 'training'), self.train_batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader( Dataset(self.database, 'testing'), self.test_batch_size, shuffle=True)

        self.create_model()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def run(self):

        for epoch in range(10):

            print("Beginning epoch " + str(epoch))

            self.epoch_train()
            self.save_checkpoint()
            self.epoch_test()

    def save_checkpoint(self):
        number = math.floor(time.time())
        filename = "model_epoch_{}.pth".format(number)
        torch.save(self.model, filename)
        print("Checkpoint saved to {}".format(filename))

    def epoch_test(self):

        num_iterations = 0
        losses = list()
        successes = list()

        with torch.no_grad():
            for x, yref in self.test_loader:

                x = x.to(self.device)
                yref = yref.to(self.device)

                y = self.model(x)
                loss = self.criterion(y, yref)

                losses.append(loss.item())
                successes += [ int(x) for x in ( y.argmax(1) == yref ) ]

                print("TestIteration({}): loss={} proba={}".format(num_iterations, loss.item(), numpy.exp(-loss.item())))

                num_iterations += 1

        proba = numpy.exp( -(sum(losses) / len(losses)) )
        print("Proba on testing set: {}".format(proba))
        print("Percentage of correct classifications: {}".format(sum(successes)/float(len(successes))))

    def epoch_train(self):

        num_iterations = 0
        losses = list()
        successes = list()

        for x, yref in self.train_loader:

            x = x.to(self.device)
            yref = yref.to(self.device)

            self.optimizer.zero_grad()
            y = self.model(x)
            loss = self.criterion(y, yref)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            successes += [ int(x) for x in ( y.argmax(1) == yref ) ]

            print("TrainIteration({}): loss={} proba={}".format(num_iterations, loss.item(), numpy.exp(-loss.item())))

            num_iterations += 1

        proba = numpy.exp( -(sum(losses) / len(losses)) )
        print("Proba on training set: {}".format(proba))
        print("Percentage of correct classifications: {}".format(sum(successes)/float(len(successes))))

    def create_model(self):

        A = [ re.match("^model_epoch_([0-9]*).pth$", f) for f in os.listdir(".") ]
        B = [ (int(m.group(1)), m.group(0)) for m in A if m ]

        if B:
            B.sort(reverse=True)
            filename = B[0][1]
            print("Loading checkpoint {}".format(filename))
            model = torch.load(filename)
            model.eval()
        else:
            model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
            for name, param in model.named_parameters():
                if "bn" not in name:
                    param.requires_grad = False

            M = 800
            N = len(self.database.class_ids)

            model.fc = torch.nn.Sequential(
                torch.nn.Linear(model.fc.in_features, M),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(M, N))

        #return model
        self.model = model.to(self.device)

        #########""
        #m = torch.jit.script(model)
        #m.save("model.pt")
        #exit(0)
        #########""

if __name__ == '__main__':

    if not torch.cuda.is_available():
        print("CUDA unavailable!")
        exit(1)

    if len(sys.argv) != 2:
        print("Bad command line!")
        exit(1)

    database = Database(sys.argv[1])

    #database.create_dataset("training", 1000)
    #database.create_dataset("testing", 170)

    trainer = Trainer(database)
    trainer.run()

