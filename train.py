import cv2
import random
import numpy
import sqlite3
import sys
import os
import os.path
import torch
import torchvision.transforms

class Dataset(torch.utils.data.Dataset):

    def __init__(self, engine, dataset_name):
        self.engine = engine
        self.dataset_name = dataset_name

        cur = self.engine.db.cursor()
        cur.execute("SELECT filename, class_id FROM samples WHERE id in (SELECT sample_id FROM memberships WHERE dataset_id IN (SELECT id FROM datasets WHERE name=?))", (self.dataset_name,))
        self.table = cur.fetchall()
        cur.close()
        del cur

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):

        item = self.table[i]

        image_filename = os.path.join(self.engine.root_directory, item[0])
        im = cv2.imread(image_filename)
        im = numpy.stack( (im[:,:,0], im[:,:,1], im[:,:,2]) )
        im = torch.Tensor(im)
        im = self.normalize(im)

        index = self.engine.class_id_to_index[item[1]]

        return (im, index)

class Engine:

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

def create_model():

    model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
    for name, param in model.named_parameters():
        if "bn" not in name:
            param.requires_grad = False

    M = 800
    N = len(engine.class_ids)

    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, M),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(M, N))

    return model

if __name__ == '__main__':

    if not torch.cuda.is_available():
        print("CUDA unavailable!")
        exit(1)

    if len(sys.argv) != 2:
        print("Bad command line!")
        exit(1)

    engine = Engine(sys.argv[1])

    #engine.create_dataset("training", 1000)
    #engine.create_dataset("testing", 170)

    train_batch_size = 64
    #test_batch_size = 64

    train_loader = torch.utils.data.DataLoader( Dataset(engine, 'training'), train_batch_size, shuffle=True)
    #test_loader = torch.utils.data.DataLoader( Dataset(engine, 'testing'), test_batch_size, shuffle=True)

    model = create_model()

    model.to('cuda')

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(20):
        print("Beginning epoch " + str(epoch))
        for x, yref in train_loader:
            optimizer.zero_grad()
            y = model(x)
            loss = criterion(y, yref)
            loss.backward()
            optimizer.step()
            print(".")

