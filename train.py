import cv2
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

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):

        item = self.table[i]

        image_filename = os.path.join(self.database.root_directory, item[0])
        im = cv2.imread(image_filename)
        im = numpy.stack( (im[:,:,0], im[:,:,1], im[:,:,2]) )
        im = torch.Tensor(im)
        im = self.normalize(im)

        index = self.database.class_id_to_index[item[1]]

        return (im, index)

class Trainer:

    def __init__(self, database):

        self.database = database

        self.train_batch_size = 64
        self.test_batch_size = 64

        self.device = 'cuda'

        self.train_loader = torch.utils.data.DataLoader( Dataset(self.database, 'training'), self.train_batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader( Dataset(self.database, 'testing'), self.test_batch_size, shuffle=True)

        self.create_model()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)


    def run(self):

        self.train_costs = list()
        self.test_costs = list()

        for epoch in range(20):

            print("Beginning epoch " + str(epoch))

            self.epoch_train()
            self.save_checkpoint(epoch)
            self.epoch_test()

    def save_checkpoint(self, epoch):
        filename = "model_epoch_{}.pth".format(epoch)
        torch.save(self.model, filename)
        print("Checkpoint saved to {}".format(filename))

    def epoch_test(self):
        #  TODO
        pass

    def epoch_train(self):

        epoch_loss = 0.0
        num_iterations = 0
        for x, yref in self.train_loader:

            x = x.to(self.device)
            yref = yref.to(self.device)

            self.optimizer.zero_grad()
            y = self.model(x)
            loss = self.criterion(y, yref)
            loss.backward()
            epoch_loss += loss.item()
            self.optimizer.step()
            print("Iteration({}): {}".format(num_iterations, loss.item()))
            self.train_costs.append(loss.item())
            num_iterations += 1

        print("Epoch ended. Loss = " + str(epoch_loss) + ".")

    def create_model(self):

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

