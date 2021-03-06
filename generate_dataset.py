import re
import numpy
import sqlite3
import cv2
import sys
import os.path

couleurs = set(['trefle', 'carreau', 'coeur', 'pique'])
valeurs = set(['as', 'roi', 'dame', 'valet', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

#to_skip = [
#    ('5', 'carreau'),
#    ('8', 'carreau'),
#    ('as', 'carreau'),
#    ('as', 'coeur'),
#    ('as', 'pique'),
#    ('as', 'trefle'),
#    ('dame', 'carreau'),
#    ('dame', 'coeur'),
#    ('dame', 'pique'),
#    ('dame', 'trefle'),
#    ('roi', 'carreau'),
#    ('roi', 'coeur'),
#    ('roi', 'pique'),
#    ('roi', 'trefle'),
#    ('valet', 'carreau'),
#    ('valet', 'coeur'),
#    ('valet', 'pique'),
#    ('valet', 'trefle') ]

to_skip = []

class SimpleFilter:

    def __init__(self, flipx, flipy):
        self.flipx = flipx
        self.flipy = flipy
        self.refshape = (512, 288)

    def __call__(self, input_):

        assert(input_.dtype == numpy.uint8)

        if input_.shape[1] > input_.shape[0]:
            a = numpy.rot90(input_)
        else:
            a = input_

        if a.shape == self.refshape:
            b = a
        else:
            b = cv2.resize(a, (self.refshape[1], self.refshape[0]))

        if self.flipx and self.flipy:
            c = cv2.flip(b, -1)
        elif self.flipx:
            c = cv2.flip(b, 0)
        elif self.flipy:
            c = cv2.flip(b, 1)
        else:
            c = b

        gamma = 2.0 ** ( 2.0*numpy.random.random() - 1.0 )

        lut = numpy.empty((1,256), numpy.uint8)
        for i in range(256):
            lut[0,i] = numpy.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        d = cv2.LUT(c, lut)

        return d

filters = [
    SimpleFilter(False, False),
    SimpleFilter(True, True)]

class DatasetGenerator:

    def init_db(self):

        fname = os.path.join(self.dataset_directory, 'db.sqlite')

        self.db  = sqlite3.connect(fname)

        self.db.execute("CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY, class_id INTEGER, filename TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS classes(id INTEGER PRIMARY KEY, suit TEXT, rank TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS datasets(id INTEGER PRIMARY KEY, name TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS memberships(id INTEGER PRIMARY KEY, sample_id INTEGER, dataset_id INTEGER)")

    def run(self, videos_directory):

        self.videos_directory = videos_directory #os.path.join(root_directory, '00_videos')
        self.dataset_directory = os.getcwd() #os.path.join(root_directory, '01_dataset')

        self.init_db()

        for f in os.listdir(self.videos_directory):
            m = re.match("^[0-9][0-9]_([a-zA-Z0-9]+)_([a-zA-Z0-9]+).mp4$", f)
            if m:
                valeur = m.group(1)
                enseigne = m.group(2)
                if valeur in valeurs and enseigne in couleurs and (valeur, enseigne) not in to_skip:
                    self.process_video(f, valeur, enseigne)
                else:
                    print("Skipping:", valeur, enseigne)

        self.db.close()

    def process_video(self, video_file, valeur, enseigne):

        print("Processing " + str(video_file) + "...")

        # create directory.

        level1_dir = os.path.join(self.dataset_directory, valeur + '_' + enseigne)
        if os.path.isdir(level1_dir):
            print("Warning: already exists:", level1_dir)
        else:
            os.mkdir(level1_dir)

        # open video.

        v = cv2.VideoCapture()
        v.open( os.path.join(self.videos_directory, video_file) )

        # create DB record

        c = self.db.cursor()
        c.execute("INSERT INTO classes(suit, rank) VALUES(?,?)", (enseigne, valeur))
        class_id = c.lastrowid
        c.close()

        export_counter = 0
        frame_counter = 0
        period = 6

        go_on = True
        while go_on:

            go_on, frame = v.read()

            if go_on and frame_counter % period == 0:
                for f in filters:
                    filtered = f(frame)
                    output_filename = os.path.join(level1_dir, str(export_counter).zfill(6) + ".png")
                    cv2.imwrite(output_filename, filtered)

                    relpath = os.path.relpath(output_filename, start=self.dataset_directory)

                    c = self.db.cursor()
                    c.execute("INSERT INTO samples(class_id, filename) VALUES(?,?)", (class_id, relpath))
                    c.close()

                    cv2.imshow("", filtered)
                    cv2.waitKey(20)

                    export_counter += 1
            frame_counter += 1

        self.db.commit()

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Bad command line!")
        exit(1)

    g = DatasetGenerator()
    g.run(sys.argv[1])

